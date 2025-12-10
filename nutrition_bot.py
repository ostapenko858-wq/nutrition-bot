import os
import io
import logging
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, Any, List

import base64
import json

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

from openai import OpenAI

# ========= НАСТРОЙКИ И КЛИЕНТЫ =========

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_TOKEN в переменных окружения.")
if not OPENAI_API_KEY:
    raise RuntimeError("Не найден OPENAI_API_KEY в переменных окружения.")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ========= ПРОСТОЕ ХРАНИЛИЩЕ В ПАМЯТИ =========

# user_states:
#   step
#   pending_meal
#   awaiting_new_day_answer
#   active_day
#   awaiting_history_day
#   awaiting_correction
user_states: Dict[int, Dict[str, Any]] = {}

# анкета и цели (включая КБЖУ)
user_profiles: Dict[int, Dict[str, Any]] = {}

# user_food_log: user -> day -> [ {description, calories, protein, fat, carbs}, ... ]
user_food_log: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(
    lambda: defaultdict(list)
)

# для психолога
user_chat_history: Dict[int, List[Dict[str, str]]] = defaultdict(list)

PROFILE_FILE = "profiles.json"


# ========= ЗАГРУЗКА/СОХРАНЕНИЕ ПРОФИЛЕЙ =========

def load_profiles():
    global user_profiles
    if not os.path.exists(PROFILE_FILE):
        return
    try:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        user_profiles = {int(uid): profile for uid, profile in data.items()}
        logger.info("Загружено профилей пользователей: %d", len(user_profiles))
    except Exception as e:
        logger.exception("Ошибка при загрузке профилей: %s", e)
        user_profiles = {}


def save_profiles():
    try:
        data = {str(uid): profile for uid, profile in user_profiles.items()}
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Ошибка при сохранении профилей: %s", e)


load_profiles()


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========

def parse_float(text: str) -> float:
    return float(text.replace(",", ".").strip())


def calculate_bmr(weight: float, height: float, age: int, sex: str) -> float:
    """
    Формула Миффлина–Сан Жеора:
    Мужчины: BMR = 10*вес + 6.25*рост - 5*возраст + 5
    Женщины: BMR = 10*вес + 6.25*рост - 5*возраст - 161
    """
    sex = sex.lower()
    if sex.startswith("ж"):
        s = -161
    else:
        s = 5
    return 10 * weight + 6.25 * height - 5 * age + s


def infer_goal_type(current_weight: float, target_weight: float) -> str:
    """
    Возвращает:
      "loss" - если цель похудеть
      "gain" - если цель набрать
      "maintain" - если цель примерно удержание
    """
    diff = target_weight - current_weight
    if diff < -1:
        return "loss"
    elif diff > 1:
        return "gain"
    else:
        return "maintain"


def plan_calories_and_deficit(
    bmr: float, current_weight: float, target_weight: float, months_to_goal: float
) -> Dict[str, Any]:
    """
    Планирует колораж и дефицит/профицит с учётом цели.
    Для похудения:
        - считаем сколько кг нужно сбросить
        - 1 кг жира ~ 7700 ккал
        - делим на дни, получаем нужный дефицит
        - ограничиваем его 300–700 ккал/день
        - если цель слишком агрессивная — считаем реалистичный срок
    Для набора — делаем профицит 200–400 ккал/день.
    Для удержания — дефицит 0.
    """
    goal_type = infer_goal_type(current_weight, target_weight)
    maintenance = bmr * 1.3  # очень грубая оценка TDEE при низкой-умеренной активности

    months_to_goal = max(months_to_goal, 0.5)
    days_planned = max(int(months_to_goal * 30), 30)

    daily_deficit = 0.0
    realistic_months = months_to_goal
    plan_comment = ""

    if goal_type == "loss":
        kg_to_lose = current_weight - target_weight
        total_deficit = kg_to_lose * 7700  # ккал
        required_deficit = total_deficit / days_planned
        min_deficit = 300.0
        max_deficit = 700.0

        if required_deficit > max_deficit:
            # цель слишком агрессивная, делаем максимально безопасный дефицит
            daily_deficit = max_deficit
            realistic_days = total_deficit / daily_deficit
            realistic_months = realistic_days / 30.0
            plan_comment = (
                f"Чтобы снизить вес на ~{kg_to_lose:.1f} кг безопасно, "
                f"тебе понадобится примерно {realistic_months:.1f} месяцев при "
                f"дефиците около {daily_deficit:.0f} ккал/день. "
                f"Это больше, чем выбранные {months_to_goal:.1f} месяцев, "
                "но так гораздо мягче и безопаснее."
            )
        elif required_deficit < min_deficit:
            daily_deficit = min_deficit
            realistic_days = total_deficit / daily_deficit
            realistic_months = realistic_days / 30.0
            plan_comment = (
                f"Твоя цель довольно мягкая: чтобы уйти с {current_weight:.1f} до "
                f"{target_weight:.1f} кг, при дефиците ~{daily_deficit:.0f} ккал/день "
                f"понадобится примерно {realistic_months:.1f} месяцев. "
                "Можно идти и быстрее, но так комфортнее."
            )
        else:
            daily_deficit = required_deficit
            realistic_months = months_to_goal
            plan_comment = (
                f"Чтобы уйти с {current_weight:.1f} до {target_weight:.1f} кг за "
                f"{months_to_goal:.1f} месяцев, нужен дефицит примерно "
                f"{daily_deficit:.0f} ккал/день. Это в разумных пределах."
            )

        target_calories = max(maintenance - daily_deficit, 1200.0)

    elif goal_type == "gain":
        # Набор — аккуратный профицит
        kg_to_gain = target_weight - current_weight
        total_surplus = kg_to_gain * 7700
        # делаем мягкий профицит
        daily_deficit = -300.0  # отрицательное число = профицит
        realistic_days = total_surplus / abs(daily_deficit)
        realistic_months = realistic_days / 30.0
        target_calories = maintenance + abs(daily_deficit)
        plan_comment = (
            f"Для набора веса с {current_weight:.1f} до {target_weight:.1f} кг "
            f"я заложу профицит около 300 ккал/день. "
            f"Это займёт примерно {realistic_months:.1f} месяцев."
        )
    else:  # maintain
        daily_deficit = 0.0
        target_calories = maintenance
        realistic_months = months_to_goal
        plan_comment = (
            "Цель по весу около текущего — будем удерживать вес вокруг этой точки "
            "без выраженного дефицита или профицита."
        )

    return {
        "goal_type": goal_type,
        "maintenance_calories": maintenance,
        "target_calories": target_calories,
        "daily_deficit": daily_deficit,  # отрицательное число = профицит
        "realistic_months": realistic_months,
        "plan_comment": plan_comment,
    }


def calculate_macro_targets(
    current_weight: float,
    target_weight: float,
    target_calories: float,
    goal_type: str,
) -> Dict[str, float]:
    """
    Рекомендации по КБЖУ, граммы в день.
    Для похудения и набора ориентируемся на целевой вес.
    """
    if goal_type == "loss":
        base_w = target_weight
        protein_g = 1.8 * base_w
        fat_g = 0.8 * base_w
    elif goal_type == "gain":
        base_w = target_weight
        protein_g = 2.0 * base_w
        fat_g = 1.0 * base_w
    else:  # maintain
        base_w = current_weight
        protein_g = 1.6 * base_w
        fat_g = 1.0 * base_w

    cal_from_protein_fat = protein_g * 4 + fat_g * 9
    carbs_cal = max(target_calories - cal_from_protein_fat, target_calories * 0.3)
    carbs_g = carbs_cal / 4.0

    return {
        "target_protein": protein_g,
        "target_fat": fat_g,
        "target_carbs": carbs_g,
    }


def today_str() -> str:
    return date.today().isoformat()


def is_correction_command(text: str) -> bool:
    """
    Определяем, что пользователь хочет пересчитать/уточнить еду.
    Реагируем на фразы типа:
    - исправь / исправление
    - уточни / уточнение
    - скорректируй / корректировка
    - поправь
    - пересчитай
    - ты ошибся / не то / не так
    """
    lowered = text.strip().lower()
    keywords = [
        "исправ",
        "уточн",
        "скоррект",
        "поправ",
        "пересчит",
        "не то",
        "не так",
        "ошиб",
    ]
    return any(kw in lowered for kw in keywords)


# ========= OPENAI: АНАЛИЗ ФОТО ЕДЫ / ЭТИКЕТКИ =========

async def analyze_food_image(image_bytes: bytes):
    """
    Анализирует фото еды ИЛИ фото этикетки через GPT-4.1 (vision) и возвращает:
      description: короткое описание
      items: список [{name, grams, calories, protein, fat, carbs}, ...]
      total_calories, total_protein, total_fat, total_carbs
      need_clarification: bool
      clarification_question: str
    """
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_b64}"

    system_prompt = """
Ты — ассистент-нутрициолог. Пользователь прислал фото своей еды или этикетки продукта.

Твоя задача:
1. Определи, что на картинке:
   - тарелка с едой / напитком
   - или этикетка/упаковка с таблицей КБЖУ.
2. Если это ТАРЕЛКА / ЕДА:
   - Определи основные блюда и напитки.
   - Для каждого оцени примерный размер порции в граммах и калорийность.
   - Также оцени белки, жиры, углеводы для каждого компонента.
   - Ориентируйся на типичные значения (курица, рис, макароны, картофель, салаты, сладкие напитки и т.п.).
3. Если это ЭТИКЕТКА:
   - Прочитай таблицу КБЖУ.
   - Определи калории, белки, жиры, углеводы хотя бы:
       • на 100 г/мл
       • и, если указано, на порцию (батончик, стакан и т.п.).
   - Если понятно, какая порция съедена (1 батончик, стандартная кружка и т.п.) —
     оцени КБЖУ именно этой порции.
4. Посчитай суммарную калорийность, белки, жиры, углеводы по всему тому,
   что пользователь, скорее всего, съел/выпил.
5. Коротко опиши, что ты видишь на фото (один абзац).

Очень важно:
— Значения калорий и КБЖУ должны быть реалистичными, без завышения/занижения в разы.
— Если порции или состав явно неочевидны, ОБЯЗАТЕЛЬНО:
    * установи "need_clarification": true
    * сформулируй понятный вопрос пользователю в "clarification_question"
      (например: "Сколько примерно грамм мяса и гарнира?" или
      "Сколько батончиков/порций ты съел?").
— Пользователь потом ответит на этот вопрос текстом или голосом, и по этому ответу
  мы пересчитаем всё ещё раз.

Ответ ДОЛЖЕН быть строго в формате JSON (без текста до или после):

{
  "description": "краткое описание по-русски",
  "items": [
    {
      "name": "название блюда или продукта",
      "grams": 150,
      "calories": 350,
      "protein": 25,
      "fat": 10,
      "carbs": 40
    }
  ],
  "total_calories": 350,
  "total_protein": 25,
  "total_fat": 10,
  "total_carbs": 40,
  "need_clarification": false,
  "clarification_question": ""
}

Если на фото совсем нет еды и нет этикетки с КБЖУ, тогда верни:

{
  "description": "на этом фото не видно еды или этикетки с КБЖУ",
  "items": [],
  "total_calories": 0,
  "total_protein": 0,
  "total_fat": 0,
  "total_carbs": 0,
  "need_clarification": false,
  "clarification_question": ""
}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Проанализируй это фото (еда или этикетка) и дай ответ строго в нужном JSON формате.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        temperature=0.25,
        max_tokens=800,
    )

    msg_content = completion.choices[0].message.content

    if isinstance(msg_content, str):
        raw_text = msg_content.strip()
    else:
        texts = []
        for part in msg_content:
            try:
                if getattr(part, "type", None) == "text":
                    texts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            except Exception:
                continue
        raw_text = "\n".join(texts).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Не удалось найти JSON в ответе модели: {raw_text!r}")
        json_str = raw_text[start: end + 1]
        data = json.loads(json_str)

    description = str(data.get("description", "")).strip()
    items_raw = data.get("items", [])
    total_calories = float(data.get("total_calories", 0) or 0)
    total_protein = float(data.get("total_protein", 0) or 0)
    total_fat = float(data.get("total_fat", 0) or 0)
    total_carbs = float(data.get("total_carbs", 0) or 0)
    need_clarification = bool(data.get("need_clarification", False))
    clarification_question = str(data.get("clarification_question", "") or "").strip()

    items = []
    for item in items_raw:
        name = str(item.get("name", "Блюдо"))
        calories = float(item.get("calories", 0) or 0)
        grams = float(item.get("grams", 0) or 0)
        protein = float(item.get("protein", 0) or 0)
        fat = float(item.get("fat", 0) or 0)
        carbs = float(item.get("carbs", 0) or 0)
        items.append(
            {
                "name": name,
                "grams": grams,
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
            }
        )

    if not total_calories and items:
        total_calories = sum(i["calories"] for i in items)
    if not total_protein and items:
        total_protein = sum(i["protein"] for i in items)
    if not total_fat and items:
        total_fat = sum(i["fat"] for i in items)
    if not total_carbs and items:
        total_carbs = sum(i["carbs"] for i in items)

    return (
        description,
        items,
        total_calories,
        total_protein,
        total_fat,
        total_carbs,
        need_clarification,
        clarification_question,
    )


# ========= OPENAI: ПЕРЕСЧЁТ ПО ТЕКСТУ =========

async def analyze_food_text(text_description: str):
    """
    Пользователь описывает еду текстом (исправляет бота или уточняет порции).
    Возвращает description, items, total_calories, total_protein, total_fat, total_carbs.
    """
    system_prompt = """
Ты — ассистент-нутрициолог.
Пользователь словами описывает, что именно он съел и сколько примерно по граммам или штукам.

Иногда он пишет не полный список еды, а только исправление к уже распознанному фото:
например: "ты ошибся, это была говядина 120 г и гречка 150 г, без соуса".
Твоя задача — понять ОКОНЧАТЕЛЬНЫЙ состав приёма пищи и посчитать КБЖУ именно для него.

1. Разбей описание на отдельные блюда/компоненты.
2. Для каждого оцени массу в граммах (если явно не указана — логично предположи)
   и калорийность, а также белки, жиры, углеводы.
3. Посчитай суммарную калорийность и КБЖУ.
4. Кратко опиши приём пищи.

Верни строго JSON:

{
  "description": "краткое описание по-русски",
  "items": [
    {
      "name": "название блюда",
      "grams": 150,
      "calories": 350,
      "protein": 25,
      "fat": 10,
      "carbs": 40
    }
  ],
  "total_calories": 350,
  "total_protein": 25,
  "total_fat": 10,
  "total_carbs": 40
}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_description},
        ],
        temperature=0.25,
        max_tokens=700,
    )

    raw_text = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Не удалось найти JSON в ответе модели: {raw_text!r}")
        json_str = raw_text[start: end + 1]
        data = json.loads(json_str)

    description = str(data.get("description", "")).strip()
    items_raw = data.get("items", [])
    total_calories = float(data.get("total_calories", 0) or 0)
    total_protein = float(data.get("total_protein", 0) or 0)
    total_fat = float(data.get("total_fat", 0) or 0)
    total_carbs = float(data.get("total_carbs", 0) or 0)

    items = []
    for item in items_raw:
        name = str(item.get("name", "Блюдо"))
        calories = float(item.get("calories", 0) or 0)
        grams = float(item.get("grams", 0) or 0)
        protein = float(item.get("protein", 0) or 0)
        fat = float(item.get("fat", 0) or 0)
        carbs = float(item.get("carbs", 0) or 0)
        items.append(
            {
                "name": name,
                "grams": grams,
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
            }
        )

    if not total_calories and items:
        total_calories = sum(i["calories"] for i in items)
    if not total_protein and items:
        total_protein = sum(i["protein"] for i in items)
    if not total_fat and items:
        total_fat = sum(i["fat"] for i in items)
    if not total_carbs and items:
        total_carbs = sum(i["carbs"] for i in items)

    return description, items, total_calories, total_protein, total_fat, total_carbs


# ========= OPENAI: РАСПОЗНАВАНИЕ ГОЛОСА =========

async def transcribe_voice(voice_bytes: bytes) -> str:
    audio_file = io.BytesIO(voice_bytes)
    audio_file.name = "voice.ogg"

    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return (transcription.text or "").strip()
    except Exception as e:
        logger.exception("Ошибка при распознавании голоса: %s", e)
        return ""


# ========= СОБЕСЕДНИК ПО ПИТАНИЮ =========

async def chat_with_coach(user_id: int, user_text: str) -> str:
    history = user_chat_history[user_id]

    system_instruction = """
Ты — доброжелательный и поддерживающий чат-бот по питанию и психологии питания.

Твоя зона:
— привычки в еде;
— отношения с едой;
— эмоции вокруг питания (стресс, переедание, «сорвалась», чувство вины и т.п.);
— мягкая поддержка здорового, НЕ экстремального подхода к питанию.

Как отвечать:
— Говори по-русски, простым и тёплым языком.
— Не стыди и не критикуй.
— Нормализуй: переедания и «лишние кусочки торта» иногда случаются со всеми.
— Помогай увидеть контекст (стресс, недосып, цикл, пропуски приёмов пищи, скука и т.п.).
— Помогай выбрать мягкий следующий шаг.
— Не давай жёстких диет, детоксов, голодовок.
— Не ставь диагнозов и не назначай лекарства.
— Если говорят о самоповреждении или суициде, мягко предложи обратиться к врачу/психотерапевту и в экстренные службы.

Если тебя уводят в темы вне еды/тела/эмоций вокруг питания
(политика, техника, финансы и т.п.), вежливо верни разговор к питанию.
"""

    messages = [{"role": "system", "content": system_instruction}]

    for m in history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )

    reply = completion.choices[0].message.content.strip()

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    user_chat_history[user_id] = history[-20:]

    return reply


# ========= ИСПРАВЛЕНИЕ / УТОЧНЕНИЕ ПОСЛЕДНЕГО ФОТО =========

async def handle_correction(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    state = user_states.get(user_id, {})
    pending = state.get("pending_meal")

    if not pending:
        await update.message.reply_text(
            "Пока нечего пересчитывать 🙂\n"
            "Сначала отправь фото еды или этикетки, я посчитаю, а потом можно будет уточнить или исправить."
        )
        return

    original = text.strip()
    lowered = original.lower()

    # пробуем вырезать слово-триггер, если оно есть
    clean_text = original
    for kw in ["исправ", "уточн", "скоррект", "поправ", "пересчит", "не то", "не так", "ошиб"]:
        idx = lowered.find(kw)
        if idx != -1:
            clean_text = original[idx + len(kw):].strip()
            break

    # если ещё есть двоеточие — убираем всё до него
    if ":" in clean_text:
        _, _, clean_text = clean_text.partition(":")
        clean_text = clean_text.strip()

    if not clean_text:
        await update.message.reply_text(
            "Опиши, пожалуйста, словами, что именно было в тарелке/порции — например:\n"
            "«там была гречка 150 г, говядина 120 г и салат из огурцов без масла»."
        )
        return

    try:
        description, items, total_cal, total_p, total_f, total_c = await analyze_food_text(clean_text)
    except Exception as e:
        logger.exception("Ошибка при пересчёте по тексту: %s", e)
        await update.message.reply_text(
            f"Не получилось пересчитать по описанию 😔\n"
            f"Ошибка OpenAI: {e}"
        )
        return

    day = pending["day"]
    count = pending["count"]
    entries = user_food_log[user_id][day]

    # удаляем старые записи, связанные с этим фото
    if count > 0 and len(entries) >= count:
        del entries[-count:]

    before_len = len(entries)
    for item in items:
        entries.append(
            {
                "description": item.get("name", "Блюдо"),
                "calories": float(item.get("calories", 0) or 0),
                "protein": float(item.get("protein", 0) or 0),
                "fat": float(item.get("fat", 0) or 0),
                "carbs": float(item.get("carbs", 0) or 0),
            }
        )
    added_count = len(entries) - before_len

    state["pending_meal"] = {"day": day, "count": added_count}
    user_states[user_id] = state

    lines = []
    lines.append("Хорошо, давай считать так:\n")

    if description:
        lines.append(f"📋 {description}\n")

    if items:
        for item in items:
            name = item.get("name", "Блюдо")
            grams = item.get("grams", 0)
            cal = item.get("calories", 0)
            p = item.get("protein", 0)
            f = item.get("fat", 0)
            c = item.get("carbs", 0)
            if grams:
                lines.append(
                    f"• {name} — ~{round(grams)} г, "
                    f"~{round(cal)} ккал, Б {round(p)} г, Ж {round(f)} г, У {round(c)} г"
                )
            else:
                lines.append(
                    f"• {name} — ~{round(cal)} ккал, Б {round(p)} г, Ж {round(f)} г, У {round(c)} г"
                )
        lines.append(
            f"\n🔢 Примерно всего: *{round(total_cal)} ккал*, "
            f"Б *{round(total_p)} г*, Ж *{round(total_f)} г*, У *{round(total_c)} г*"
        )
    else:
        lines.append("Не смогла распознать блюда по описанию 😔")

    lines.append(
        "\nЯ обновила записи за этот день с учётом этого уточнения/исправления."
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
    )


# ========= ИСТОРИЯ ЗА ДЕНЬ / НЕДЕЛЮ (без изменений логики) =========
# ... (оставляем те же send_day_history, week_history, day_history_command,
# week_history_command, что были у тебя в рабочей версии — они
# не зависят от новых расчётов и моделей)

# Чтобы не раздувать ответ ещё сильнее:
# представь, что здесь идут ТВОИ же функции send_day_history, week_history,
# week_history_command, day_history_command — их можно оставить без изменений.


# ========= ОБРАБОТКА ТЕКСТА И НОВЫЙ ОПРОС =========

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {"step": None})
    step = state.get("step")
    lowered = text.strip().lower()

    # если ждём ответ "новый день?"
    if state.get("awaiting_new_day_answer"):
        if lowered.startswith("да"):
            new_day = today_str()
            state["active_day"] = new_day
            state["awaiting_new_day_answer"] = False
            await update.message.reply_text(
                "Окей, считаем, что начался новый день 🌞\n"
                "Теперь следующий приём пищи я буду записывать как новый день."
            )
            return
        elif lowered.startswith("нет"):
            state["awaiting_new_day_answer"] = False
            await update.message.reply_text(
                "Хорошо, продолжаем считать, что это тот же день 🌙"
            )
            return

    # если ждём дату для истории — логика как была, опускаю тут ради длины

    # ===== коррекция после фото =====
    if state.get("awaiting_correction") and state.get("pending_meal"):
        # любое следующее сообщение после фото считаем уточнением/исправлением
        state["awaiting_correction"] = False
        await handle_correction(update, context, text)
        return

    # ЯВНЫЕ фразы "исправь/уточни/скорректируй..."
    if is_correction_command(text):
        await handle_correction(update, context, text)
        return

    # ===== если ещё нет профиля =====
    if step is None and user_id not in user_profiles:
        await update.message.reply_text(
            "Я пока с тобой не знаком 😅\n"
            "Нажми /start, чтобы пройти короткую анкету с расчётом BMR и цели по весу."
        )
        return

    # ===== новая анкета =====
    if step == "ask_name":
        name = text.strip()
        user_profiles[user_id] = {"name": name}
        save_profiles()
        state["step"] = "ask_age"
        await update.message.reply_text(
            f"Рада знакомству, {name}! 🎉\nСколько тебе полных лет?"
        )
        return

    if step == "ask_age":
        try:
            age = int(text.strip())
            if age <= 0 or age > 120:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Возраст должен быть целым числом (например, 28). Попробуй ещё раз 🙂"
            )
            return

        user_profiles[user_id]["age"] = age
        save_profiles()
        state["step"] = "ask_sex"
        await update.message.reply_text(
            "Укажи, пожалуйста, пол (м/ж). Это нужно для более точного расчёта метаболизма."
        )
        return

    if step == "ask_sex":
        sex = text.strip().lower()
        if not (sex.startswith("м") or sex.startswith("ж")):
            await update.message.reply_text(
                "Напиши, пожалуйста, просто 'м' или 'ж' 🙂"
            )
            return
        user_profiles[user_id]["sex"] = sex
        save_profiles()
        state["step"] = "ask_weight"
        await update.message.reply_text(
            "Отлично!\nТеперь напиши свой ТЕКУЩИЙ вес в килограммах (например, 70.5)."
        )
        return

    if step == "ask_weight":
        try:
            weight = parse_float(text)
            if weight <= 0 or weight > 400:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Вес должен быть числом в килограммах (например, 68 или 68.5). Попробуй ещё раз."
            )
            return

        user_profiles[user_id]["weight"] = weight
        save_profiles()
        state["step"] = "ask_height"
        await update.message.reply_text(
            "Принято 👍\nТеперь укажи рост в сантиметрах (например, 170)."
        )
        return

    if step == "ask_height":
        try:
            height = parse_float(text)
            if height <= 0 or height > 250:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Рост должен быть числом в сантиметрах (например, 165). Попробуй ещё раз."
            )
            return

        user_profiles[user_id]["height"] = height
        save_profiles()
        state["step"] = "ask_target_weight"
        await update.message.reply_text(
            "Теперь скажи, до какого веса ты бы хотела/хотел прийти? (в килограммах)\n"
            "Например: 60"
        )
        return

    if step == "ask_target_weight":
        try:
            target_weight = parse_float(text)
            if target_weight <= 0 or target_weight > 400:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Целевой вес тоже должен быть числом в килограммах 🙂 Попробуй ещё раз."
            )
            return

        user_profiles[user_id]["target_weight"] = target_weight
        save_profiles()
        state["step"] = "ask_months_to_goal"
        await update.message.reply_text(
            "За сколько МЕСЯЦЕВ примерно ты хотела/хотел бы прийти к этому весу?\n"
            "Например: 6"
        )
        return

    if step == "ask_months_to_goal":
        try:
            months = parse_float(text)
            if months <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Срок лучше указать числом в месяцах (например, 6). Попробуй ещё раз 🙂"
            )
            return

        profile = user_profiles[user_id]
        profile["months_to_goal"] = months

        # ---- расчёты BMR и плана ----
        bmr = calculate_bmr(
            weight=profile["weight"],
            height=profile["height"],
            age=profile["age"],
            sex=profile["sex"],
        )
        plan = plan_calories_and_deficit(
            bmr=bmr,
            current_weight=profile["weight"],
            target_weight=profile["target_weight"],
            months_to_goal=months,
        )

        goal_type = plan["goal_type"]
        maintenance = plan["maintenance_calories"]
        target_cal = plan["target_calories"]
        daily_def = plan["daily_deficit"]
        realistic_months = plan["realistic_months"]
        plan_comment = plan["plan_comment"]

        macro_targets = calculate_macro_targets(
            current_weight=profile["weight"],
            target_weight=profile["target_weight"],
            target_calories=target_cal,
            goal_type=goal_type,
        )

        profile["bmr"] = bmr
        profile["goal_type"] = goal_type
        profile["maintenance_calories"] = maintenance
        profile["target_calories"] = target_cal
        profile["daily_deficit"] = daily_def
        profile["realistic_months"] = realistic_months
        profile.update(macro_targets)
        save_profiles()

        state["step"] = None
        state.setdefault("active_day", today_str())

        tp = macro_targets["target_protein"]
        tf = macro_targets["target_fat"]
        tc = macro_targets["target_carbs"]

        goal_text = {
            "loss": "похудение",
            "gain": "набор веса",
            "maintain": "удержание веса",
        }[goal_type]

        deficit_line = ""
        if goal_type == "loss":
            deficit_line = f"Ориентировочный дефицит: ~{abs(round(daily_def))} ккал/день."
        elif goal_type == "gain":
            deficit_line = f"Ориентировочный профицит: ~{abs(round(daily_def))} ккал/день."

        await update.message.reply_text(
            "Готово! Вот твоя персональная схема 🔍\n\n"
            f"🧮 Формула BMR (Миффлин–Сан Жеор):\n"
            f"*BMR = 10×вес + 6.25×рост − 5×возраст + s*, где s = 5 для мужчин и −161 для женщин.\n\n"
            f"С твоими данными BMR ≈ *{round(bmr)} ккал/день* — это энергия на жизнь в покое.\n\n"
            f"Примерный расход с учётом повседневной активности (TDEE ≈ BMR × 1.3):\n"
            f"*~{round(maintenance)} ккал/день*.\n\n"
            f"Текущий вес: *{profile['weight']} кг*\n"
            f"Целевой вес: *{profile['target_weight']} кг*\n"
            f"Тип цели: *{goal_text}*\n\n"
            f"Целевой колораж на день: *~{round(target_cal)} ккал/день*.\n"
            f"{deficit_line}\n\n"
            f"{plan_comment}\n\n"
            "Рекомендации по КБЖУ на день:\n"
            f"🥩 Белки: *~{round(tp)} г*\n"
            f"🧈 Жиры: *~{round(tf)} г*\n"
            f"🥔 Углеводы: *~{round(tc)} г*\n\n"
            "Теперь ты можешь отправлять мне *фото еды или этикеток*, а я буду считать "
            "примерные калории и КБЖУ.\n"
            "После фото можешь одним сообщением голосом или текстом описать приём пищи "
            "или поправить меня — я пересчитаю.\n\n"
            "‼️ Важно: в конце дня добавляй к итогам ещё *200–300 ккал* на возможную погрешность."
            ,
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # ===== дальше — уже знакомый пользователь =====
    # здесь оставляем твою логику: история за день, неделя, /summary, чат-коуч и т.п.
    # + если текст не попадает ни в одну команду — отправляем в chat_with_coach

    try:
        reply = await chat_with_coach(user_id, text)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("Ошибка в чат-коуче: %s", e)
        await update.message.reply_text(f"Ошибка OpenAI в чате: {e}")


# ========= START / RESET, ОБРАБОТКА ФОТО/ГОЛОСА, SUMMARY =========
# Здесь можно оставить твой рабочий код, только добавить:
#  - в photo_message после успешного анализа:
#       state["pending_meal"] = {...}
#       state["awaiting_correction"] = True
#  - в сообщении к пользователю текст типа:
#       "Если хочешь уточнить или исправить, просто в следующем сообщении
#        опиши словами, что именно ты ел(а) — я пересчитаю."


# В самом конце — обычный main() с polling:

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("week", week_history_command))
    app.add_handler(CommandHandler("dayhistory", day_history_command))
    app.add_handler(MessageHandler(filters.PHOTO, photo_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    logger.info("Бот запущен (polling). Ждём сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()
