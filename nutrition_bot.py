import os
import io
import json
import base64
import logging
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

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


# =============== НАСТРОЙКИ И КЛИЕНТЫ ===============

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


# =============== ХРАНИЛИЩЕ В ПАМЯТИ ===============

# user_states:
#   step
#   active_day
#   awaiting_correction
#   pending_meal = {"day": str, "count": int, "items": [..]}
user_states: Dict[int, Dict[str, Any]] = {}

# анкета и цели + КБЖУ
user_profiles: Dict[int, Dict[str, Any]] = {}

# user_food_log: user -> day -> [ {description, calories, protein, fat, carbs}, ... ]
user_food_log: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(
    lambda: defaultdict(list)
)

# история общения с коучем
user_chat_history: Dict[int, List[Dict[str, str]]] = defaultdict(list)

# трекер воды: user -> day -> ml
user_water_log: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

PROFILE_FILE = "profiles.json"


# уровни активности
ACTIVITY_LEVELS: Dict[str, Tuple[str, float]] = {
    "1": ("минимальная активность (сидячая работа, почти без спорта)", 1.2),
    "2": ("легкая активность (1–3 нетяжёлые тренировки в неделю)", 1.375),
    "3": ("умеренная активность (3–5 тренировок в неделю)", 1.55),
    "4": ("высокая активность (6–7 тренировок в неделю)", 1.725),
    "5": ("очень высокая активность (физический труд или спорт почти каждый день)", 1.9),
}


# =============== ПРОФИЛИ ПОЛЬЗОВАТЕЛЕЙ ===============

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


# =============== УТИЛИТЫ ===============

def parse_float(text: str) -> float:
    return float(text.replace(",", ".").strip())


def today_str() -> str:
    return date.today().isoformat()


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
    diff = target_weight - current_weight
    if diff < -1:
        return "loss"
    elif diff > 1:
        return "gain"
    return "maintain"


def parse_activity_level(text: str) -> Optional[Tuple[str, float]]:
    """
    Возвращает (описание, коэффициент) или None.
    Ожидаем число 1–5 или текст, содержащий это число.
    """
    t = text.strip().lower()
    for key in ACTIVITY_LEVELS:
        if t == key or f"{key} " in t or f" {key}" in t:
            return ACTIVITY_LEVELS[key]
    for ch in t:
        if ch in ACTIVITY_LEVELS:
            return ACTIVITY_LEVELS[ch]
    return None


def plan_calories_and_deficit(
    bmr: float,
    current_weight: float,
    target_weight: float,
    months_to_goal: float,
    activity_factor: float,
) -> Dict[str, Any]:
    """
    Строим реалистичный план калорий:
    - TDEE = BMR * activity_factor
    - для похудения считаем нужный дефицит, но ограничиваем 300–700 ккал/день
    - для набора даём профицит ~300 ккал
    """
    goal_type = infer_goal_type(current_weight, target_weight)
    maintenance = bmr * activity_factor

    months_to_goal = max(months_to_goal, 0.5)
    days_planned = max(int(months_to_goal * 30), 30)

    daily_deficit = 0.0
    realistic_months = months_to_goal
    plan_comment = ""

    if goal_type == "loss":
        kg_to_lose = current_weight - target_weight
        total_deficit = kg_to_lose * 7700
        required_deficit = total_deficit / days_planned
        min_deficit = 300.0
        max_deficit = 700.0

        if required_deficit > max_deficit:
            daily_deficit = max_deficit
            realistic_days = total_deficit / daily_deficit
            realistic_months = realistic_days / 30.0
            plan_comment = (
                f"Чтобы снизить вес на ~{kg_to_lose:.1f} кг безопасно, "
                f"понадобится примерно {realistic_months:.1f} месяцев при "
                f"дефиците около {daily_deficit:.0f} ккал/день. "
                f"Это дольше, чем выбранные {months_to_goal:.1f} месяцев, "
                "но так мягче и здоровее."
            )
        elif required_deficit < min_deficit:
            daily_deficit = min_deficit
            realistic_days = total_deficit / daily_deficit
            realistic_months = realistic_days / 30.0
            plan_comment = (
                f"Цель довольно мягкая: при дефиците ~{daily_deficit:.0f} ккал/день "
                f"уйти с {current_weight:.1f} до {target_weight:.1f} кг займёт примерно "
                f"{realistic_months:.1f} месяцев."
            )
        else:
            daily_deficit = required_deficit
            realistic_months = months_to_goal
            plan_comment = (
                f"Чтобы уйти с {current_weight:.1f} до {target_weight:.1f} кг за "
                f"{months_to_goal:.1f} месяцев, нужен дефицит примерно "
                f"{daily_deficit:.0f} ккал/день. Это в комфортных пределах."
            )

        target_calories = max(maintenance - daily_deficit, 1200.0)

    elif goal_type == "gain":
        kg_to_gain = target_weight - current_weight
        total_surplus = kg_to_gain * 7700
        daily_deficit = -300.0  # профицит
        realistic_days = total_surplus / abs(daily_deficit)
        realistic_months = realistic_days / 30.0
        target_calories = maintenance + abs(daily_deficit)
        plan_comment = (
            f"Для набора веса с {current_weight:.1f} до {target_weight:.1f} кг "
            f"я заложу профицит около 300 ккал/день. "
            f"Это займёт примерно {realistic_months:.1f} месяцев."
        )
    else:
        daily_deficit = 0.0
        target_calories = maintenance
        plan_comment = (
            "Цель — удержание веса примерно на текущем уровне, без выраженного "
            "дефицита или профицита."
        )

    return {
        "goal_type": goal_type,
        "maintenance_calories": maintenance,
        "target_calories": target_calories,
        "daily_deficit": daily_deficit,
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
    Рекомендации по КБЖУ (г/день).
    """
    if goal_type == "loss":
        base_w = target_weight
        protein_g = 1.8 * base_w
        fat_g = 0.8 * base_w
    elif goal_type == "gain":
        base_w = target_weight
        protein_g = 2.0 * base_w
        fat_g = 1.0 * base_w
    else:
        base_w = current_weight
        protein_g = 1.6 * base_w
        fat_g = 1.0 * base_w

    cal_pf = protein_g * 4 + fat_g * 9
    carbs_cal = max(target_calories - cal_pf, target_calories * 0.3)
    carbs_g = carbs_cal / 4.0

    return {
        "target_protein": protein_g,
        "target_fat": fat_g,
        "target_carbs": carbs_g,
    }


def is_correction_command(text: str) -> bool:
    """
    Фраза-поправка: "исправь", "уточни", "скорректируй", "поправь",
    "пересчитай", "ты ошибся", "не то", "не так" и т.п.
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


def get_main_keyboard() -> ReplyKeyboardMarkup:
    buttons = [
        [
            KeyboardButton("📊 Саммари за день"),
            KeyboardButton("📜 История за день"),
        ],
        [
            KeyboardButton("📆 Как я ем неделю"),
            KeyboardButton("❓ Помощь"),
        ],
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True)


# =============== OPENAI: ФОТО ЕДЫ / ЭТИКЕТКИ ===============

async def analyze_food_image(image_bytes: bytes):
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_b64}"

    system_prompt = """
Ты — ассистент-нутрициолог. Пользователь прислал фото своей еды или фото этикетки с таблицей КБЖУ.

Твоя задача:

1. Определи, что на картинке:
   - тарелка с блюдом/блюдами и напитками
   - или этикетка/упаковка с таблицей КБЖУ.

2. Если это ТАРЕЛКА/ЕДА:
   - Определи основные компоненты (мясо/рыба, гарнир, салат, соусы, десерты, напитки).
   - Для каждого оцени примерный вес порции в граммах.
   - Оцени калории, белки, жиры, углеводы.
   - Используй реалистичные значения (не завышай и не занижай в разы).

3. Если это ЭТИКЕТКА:
   - Прочитай таблицу КБЖУ.
   - Определи калории, белки, жиры, углеводы хотя бы на 100 г/мл.
   - Если по этикетке понятно, какая порция съедена (1 батончик, 1 стакан и т.п.) — оцени КБЖУ этой порции.

4. Посчитай ИТОГО:
   - суммарные калории
   - суммарные белки, жиры, углеводы.

5. Кратко опиши по-русски, что именно на фото.

Очень важно:
— Если порции или состав неочевидны, ОБЯЗАТЕЛЬНО поставь "need_clarification": true
  и задай конкретный вопрос в "clarification_question"
  (например: "Сколько грамм мяса и сколько гарнира примерно?" или
   "Сколько батончиков/порций ты съел(а)?").
— Пользователь ответит на этот вопрос текстом или голосом, и по этому описанию мы пересчитаем приём пищи.

Ответ ДОЛЖЕН быть строго в формате JSON:

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

Если это совсем не похоже на еду и не похоже на этикетку с КБЖУ, верни:

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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Проанализируй это фото и верни строго JSON.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        temperature=0.25,
        max_tokens=900,
    )

    msg_content = completion.choices[0].message.content

    if isinstance(msg_content, str):
        raw_text = msg_content.strip()
    else:
        parts = []
        for part in msg_content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        raw_text = "\n".join(parts).strip()

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
        grams = float(item.get("grams", 0) or 0)
        cal = float(item.get("calories", 0) or 0)
        p = float(item.get("protein", 0) or 0)
        f = float(item.get("fat", 0) or 0)
        c = float(item.get("carbs", 0) or 0)
        items.append(
            {
                "name": name,
                "grams": grams,
                "calories": cal,
                "protein": p,
                "fat": f,
                "carbs": c,
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


# =============== OPENAI: ТЕКСТОВОЕ ОПИСАНИЕ / ИСПРАВЛЕНИЕ ===============

async def analyze_food_text(text_description: str, base_items: Optional[List[Dict[str, Any]]] = None):
    """
    Если base_items передан — это список блюд, которые бот увидел по фото.
    Тогда мы просим модель:
      «вот, что было по фото» + «вот, что написал пользователь» →
      верни ОКОНЧАТЕЛЬНЫЙ состав приёма пищи (исправленный/дополненный).
    Если base_items нет — считаем, что пользователь просто описывает приём пищи с нуля.
    """

    if base_items:
        base_lines = []
        for item in base_items:
            name = item.get("name", "Блюдо")
            grams = item.get("grams", 0)
            cal = item.get("calories", 0)
            p = item.get("protein", 0)
            f = item.get("fat", 0)
            c = item.get("carbs", 0)
            base_lines.append(
                f"- {name}, ~{round(grams)} г, ~{round(cal)} ккал, "
                f"Б {round(p)} г, Ж {round(f)} г, У {round(c)} г"
            )
        base_text = "\n".join(base_lines)
        user_prompt = f"""
Вот, что я (бот) изначально увидела по фото и посчитала как приём пищи:

{base_text}

Затем пользователь написал/сказал следующее уточнение или исправление:

\"\"\"{text_description}\"\"\"

Твоя задача:
— понять, какие блюда нужно ДОБАВИТЬ, ИЗМЕНИТЬ или УДАЛИТЬ;
— и вернуть ОКОНЧАТЕЛЬНЫЙ состав приёма пищи (полный список блюд),
  с их массой и КБЖУ.

Очень важно:
— Если пользователь пишет что-то вроде «там ещё сосиска», это значит ДОБАВИТЬ к уже указанным блюдам.
— Если пишет «не было риса» — убери рис из состава.
— Если пишет «это была говядина 120 г, а не курица» — замени курицу на говядину с новой массой.

Верни только JSON с финальным составом приёма пищи:

{{
  "description": "краткое описание по-русски",
  "items": [
    {{
      "name": "название блюда",
      "grams": 150,
      "calories": 350,
      "protein": 25,
      "fat": 10,
      "carbs": 40
    }}
  ],
  "total_calories": 350,
  "total_protein": 25,
  "total_fat": 10,
  "total_carbs": 40
}}
"""
    else:
        # старый вариант: описание с нуля
        user_prompt = f"""
Пользователь описывает приём пищи или список продуктов. 
Нужно посчитать КБЖУ.

Текст описания:

\"\"\"{text_description}\"\"\"

Сформируй список блюд и их КБЖУ и верни JSON:

{{
  "description": "краткое описание по-русски",
  "items": [
    {{
      "name": "название блюда",
      "grams": 150,
      "calories": 350,
      "protein": 25,
      "fat": 10,
      "carbs": 40
    }}
  ],
  "total_calories": 350,
  "total_protein": 25,
  "total_fat": 10,
  "total_carbs": 40
}}
"""

    system_prompt = """
Ты — ассистент-нутрициолог.
Считай калории и КБЖУ реалистично (не завышай и не занижай в разы).
Никакого текста вне JSON, только сам JSON.
"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.25,
        max_tokens=800,
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
        grams = float(item.get("grams", 0) or 0)
        cal = float(item.get("calories", 0) or 0)
        p = float(item.get("protein", 0) or 0)
        f = float(item.get("fat", 0) or 0)
        c = float(item.get("carbs", 0) or 0)
        items.append(
            {
                "name": name,
                "grams": grams,
                "calories": cal,
                "protein": p,
                "fat": f,
                "carbs": c,
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


# =============== OPENAI: ГОЛОС (WHISPER) ===============

async def transcribe_voice(voice_bytes: bytes) -> str:
    audio_file = io.BytesIO(voice_bytes)
    audio_file.name = "voice.ogg"
    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return (result.text or "").strip()
    except Exception as e:
        logger.exception("Ошибка при распознавании голоса: %s", e)
        return ""


# =============== ЧАТ КАК ПСИХОЛОГ ПО ЕДЕ ===============

async def chat_with_coach(user_id: int, user_text: str) -> str:
    history = user_chat_history[user_id]

    system_instruction = """
Ты — доброжелательный чат-бот по питанию и психологии питания.

Твоя зона:
— привычки в еде;
— эмоции вокруг еды (стресс, вина, переедание, срывы);
— мягкая поддержка без стыда и жёстких диет.

Как отвечать:
— простым, тёплым языком;
— нормализуй: «сорвался» / «съела третий кусок торта» — это не катастрофа;
— помогай видеть контекст (стресс, недосып, цикл, мало ел(а) днём, скука);
— предлагай небольшой следующий шаг, а не идеальную диету;
— не ставь диагнозов и не назначай лекарства;
— если слышишь суицидальные мысли, мягко предложи обратиться к врачу и в экстренные службы.

Если разговор уходит в темы вне еды/тела/эмоций вокруг питания (политика, техника и т.п.),
вежливо возвращай к теме отношений с едой.
"""

    messages = [{"role": "system", "content": system_instruction}]
    for msg in history[-10:]:
        messages.append(msg)
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


# =============== ИСТОРИЯ И САММАРИ ===============

def aggregate_day(user_id: int, day: str):
    entries = user_food_log[user_id].get(day, [])
    total_cal = sum(e.get("calories", 0) for e in entries)
    total_p = sum(e.get("protein", 0) for e in entries)
    total_f = sum(e.get("fat", 0) for e in entries)
    total_c = sum(e.get("carbs", 0) for e in entries)
    return total_cal, total_p, total_f, total_c, entries


async def send_day_history(update: Update, user_id: int, day: str):
    total_cal, total_p, total_f, total_c, entries = aggregate_day(user_id, day)
    if not entries:
        await update.message.reply_text("За этот день пока нет записей 🍽")
        return

    lines = [f"📜 История за {day}:"]
    for i, e in enumerate(entries, start=1):
        lines.append(
            f"{i}. {e['description']} — ~{round(e['calories'])} ккал, "
            f"Б {round(e['protein'])} г, Ж {round(e['fat'])} г, У {round(e['carbs'])} г"
        )

    lines.append(
        f"\nИтого за день: *{round(total_cal)} ккал*, "
        f"Б *{round(total_p)} г*, Ж *{round(total_f)} г*, У *{round(total_c)} г*"
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
    )


async def week_history(update: Update, user_id: int):
    today = date.today()
    lines = ["📆 Как ты ел(а) за последние 7 дней:\n"]
    profile = user_profiles.get(user_id)
    target = profile.get("target_calories") if profile else None

    for i in range(7):
        d = (today - timedelta(days=i)).isoformat()
        total_cal, total_p, total_f, total_c, _ = aggregate_day(user_id, d)
        if total_cal == 0:
            lines.append(f"{d}: нет записей")
        else:
            if target:
                diff = total_cal - target
                if abs(diff) <= target * 0.1:
                    mark = "✅ в диапазоне"
                elif diff < 0:
                    mark = "⚠️ недоел(а)"
                else:
                    mark = "⚠️ переел(а)"
                lines.append(f"{d}: ~{round(total_cal)} ккал ({mark})")
            else:
                lines.append(f"{d}: ~{round(total_cal)} ккал")

    await update.message.reply_text("\n".join(lines))


# =============== ОБРАБОТКА ИСПРАВЛЕНИЙ ПОСЛЕ ФОТО ===============

async def handle_correction(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    pending = state.get("pending_meal")

    if not pending:
        await update.message.reply_text(
            "Пока нечего пересчитывать 🙂\n"
            "Сначала отправь фото еды или этикетки, я посчитаю, а потом можно будет уточнить."
        )
        return

    original = text.strip()
    lowered = original.lower()
    clean_text = original

    for kw in ["исправ", "уточн", "скоррект", "поправ", "пересчит", "не то", "не так", "ошиб"]:
        idx = lowered.find(kw)
        if idx != -1:
            clean_text = original[idx + len(kw):].strip()
            break

    if ":" in clean_text:
        _, _, clean_text = clean_text.partition(":")
        clean_text = clean_text.strip()

    if not clean_text:
        await update.message.reply_text(
            "Опиши, пожалуйста, словами, что именно было в тарелке или порции — "
            "например: «гречка 150 г, говядина 120 г, салат без масла»."
        )
        return

    base_items = pending.get("items")  # то, что было по фото/последнему пересчёту

    try:
        desc, items, total_cal, total_p, total_f, total_c = await analyze_food_text(
            clean_text, base_items=base_items
        )
    except Exception as e:
        logger.exception("Ошибка при пересчёте по тексту: %s", e)
        await update.message.reply_text(f"Не получилось пересчитать: {e}")
        return

    day = pending["day"]
    old_count = pending["count"]
    entries = user_food_log[user_id][day]

    # удаляем старые записи, связанные с этим приёмом
    if old_count > 0 and len(entries) >= old_count:
        del entries[-old_count:]

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
    added = len(entries) - before_len
    state["pending_meal"] = {"day": day, "count": added, "items": items}
    state["awaiting_correction"] = False

    lines = ["Окей, пересчитаем так:\n"]
    if desc:
        lines.append(f"📋 {desc}\n")

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
        f"\nИтого: *{round(total_cal)} ккал*, "
        f"Б *{round(total_p)} г*, Ж *{round(total_f)} г*, У *{round(total_c)} г*"
    )
    lines.append("\nЯ обновила записи за этот день с учётом этих правок.")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


# =============== КОМАНДЫ БОТА ===============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    state = user_states.setdefault(user_id, {})
    state.setdefault("active_day", today_str())

    profile = user_profiles.get(user_id)
    if not profile:
        state["step"] = "ask_name"
        await update.message.reply_text(
            "Привет! Я нутри-бот, который поможет отслеживать питание и цели по весу 🥦\n\n"
            "Давай познакомимся и настроим тебя. Как тебя зовут?",
            reply_markup=get_main_keyboard(),
        )
        return

    # У пользователя уже есть профиль — покажем его цели
    target_cal = profile.get("target_calories")
    tp = profile.get("target_protein")
    tf = profile.get("target_fat")
    tc = profile.get("target_carbs")
    activity_desc = profile.get("activity_description")

    text_lines = [
        f"Снова привет, {profile.get('name', 'друг')} 👋",
        "Я помню твои настройки:",
    ]
    if activity_desc:
        text_lines.append(f"• Активность: {activity_desc}")
    if target_cal:
        text_lines.append(f"• Целевой колораж: ~{round(target_cal)} ккал/день")
    if tp and tf and tc:
        text_lines.append(
            f"• Белки: ~{round(tp)} г, жиры: ~{round(tf)} г, углеводы: ~{round(tc)} г в день"
        )
    text_lines.append(
        "\nПросто присылай мне фото еды или этикетки — я посчитаю примерные калории и КБЖУ.\n"
        "Если я ошибусь, сразу после фото напиши или наговори голосом, что именно ты съел(а) — я пересчитаю."
    )

    await update.message.reply_text(
        "\n".join(text_lines),
        reply_markup=get_main_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Что я умею:\n"
        "• Анкета: имя, возраст, пол, вес, рост, активность, цель по весу и срок → BMR и план.\n"
        "• На основе BMR и активности даю реалистичный план калорий и КБЖУ.\n"
        "• По фото еды или этикетки считаю калории и КБЖУ.\n"
        "• Если я неправильно распознал еду — просто напиши или скажи голосом, что на тарелке, "
        "и я пересчитаю именно этот приём пищи (исправлю/добавлю).\n"
        "• Веду историю за день и неделю.\n"
        "• Беседую про отношения с едой и эмоциональное переедание.\n\n"
        "Полезные команды:\n"
        "/start — начать или показать настройки\n"
        "/summary — итог за день\n"
        "/week — как ты ешь неделю\n"
        "/dayhistory — подробная история за сегодня\n"
        "/reset — сбросить настройки и историю для этого чата.",
        reply_markup=get_main_keyboard(),
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_states.pop(user_id, None)
    user_food_log.pop(user_id, None)
    user_chat_history.pop(user_id, None)
    user_profiles.pop(user_id, None)
    save_profiles()
    await update.message.reply_text(
        "Я всё про тебя забыла в этом чате: анкета, цели и история еды очищены 🔄\n"
        "Можно начать заново с /start.",
        reply_markup=get_main_keyboard(),
    )


async def water_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    day = state.get("active_day", today_str())
    
    try:
        if context.args:
            ml = int(context.args[0])
            user_water_log[user_id][day] += ml
            await update.message.reply_text(f"💧 Добавлено {ml} мл воды. Итого за сегодня: {user_water_log[user_id][day]} мл.")
        else:
            await update.message.reply_text("Используй: /water [количество в мл], например: /water 250")
    except ValueError:
        await update.message.reply_text("Пожалуйста, укажите количество воды числом.")

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    day = state.get("active_day", today_str())
    total_cal, total_p, total_f, total_c, entries = aggregate_day(user_id, day)
    water = user_water_log[user_id].get(day, 0)

    if not entries and water == 0:
        await update.message.reply_text(
            "За этот день у тебя пока нет записей 🍽\n"
            "Отправляй фото еды, этикетки или лог воды (/water 250).",
            reply_markup=get_main_keyboard(),
        )
        return

    profile = user_profiles.get(user_id)
    target_cal = profile.get("target_calories") if profile else None
    tp = profile.get("target_protein") if profile else None
    tf = profile.get("target_fat") if profile else None
    tc = profile.get("target_carbs") if profile else None

    lines = [f"📊 Итоги за {day}:\n"]
    lines.append(
        f"Калории: *{round(total_cal)} ккал*"
        + (f" из целевых ~{round(target_cal)}" if target_cal else "")
    )
    if tp and tf and tc:
        lines.append(
            f"Белки: *{round(total_p)} г* / цель ~{round(tp)} г\n"
            f"Жиры: *{round(total_f)} г* / цель ~{round(tf)} г\n"
            f"Углеводы: *{round(total_carbs)} г* / цель ~{round(tc)} г"
        )
    else:
        lines.append(
            f"Б: *{round(total_p)} г*, Ж: *{round(total_f)} г*, У: *{round(total_c)} г*"
        )
    
    lines.append(f"💧 Вода: *{water} мл*")

    if target_cal:
        diff = total_cal - target_cal
        if abs(diff) <= target_cal * 0.1:
            comment = "Ты молодец, сегодня ты примерно в своём диапазоне ✅"
        elif diff < 0:
            comment = (
                "Сегодня ты недоел(а) по сравнению с целевым колоражом. "
                "Иногда так бывает, но смотри, чтобы это не превращалось в жёсткое ограничение 💛"
            )
        else:
            comment = (
                "Сегодня ты вышел(ла) за пределы целевого колоража. "
                "Это не катастрофа, давай просто посмотрим, что могло повлиять: стресс, "
                "мало ел(а) днём, усталость, цикл и т.п."
            )
        lines.append("\n" + comment)

    lines.append(
        "\nИ помни: к цифрам дня можно добавить ещё 200–300 ккал на возможную погрешность."
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_keyboard(),
    )


async def day_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    day = state.get("active_day", today_str())
    await send_day_history(update, user_id, day)


async def week_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await week_history(update, user_id)


# =============== ОБРАБОТКА ФОТО / ГОЛОСА / ТЕКСТА ===============

async def photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    day = state.setdefault("active_day", today_str())

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        img_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.exception("Ошибка загрузки фото: %s", e)
        await update.message.reply_text("Не получилось скачать фото 😔 Попробуй ещё раз.")
        return

    try:
        (
            desc,
            items,
            total_cal,
            total_p,
            total_f,
            total_c,
            need_clar,
            clar_q,
        ) = await analyze_food_image(bytes(img_bytes))
    except Exception as e:
        logger.exception("Ошибка анализа фото: %s", e)
        await update.message.reply_text(
            f"Что-то пошло не так при анализе фото 😔\nОшибка OpenAI: {e}"
        )
        return

    if not items:
        await update.message.reply_text(
            "Я не вижу на этом фото еды или читаемой этикетки с КБЖУ 😔\n"
            "Попробуй сделать снимок ближе или при лучшем освещении."
        )
        return

    entries = user_food_log[user_id][day]
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

    # сохраняем исходный набор по фото для будущих исправлений
    state["pending_meal"] = {"day": day, "count": len(items), "items": items}
    state["awaiting_correction"] = True

    lines = []
    if desc:
        lines.append(f"📷 Я вижу: {desc}\n")
    else:
        lines.append("📷 Я распознала несколько компонентов:\n")

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
        f"\nИтого по этому фото: *{round(total_cal)} ккал*, "
        f"Б *{round(total_p)} г*, Ж *{round(total_f)} г*, У *{round(total_c)} г*"
    )

    if need_clar and clar_q:
        lines.append(
            "\nМне нужно чуть больше инфы, чтобы посчитать точнее:\n"
            f"❓ {clar_q}\n\n"
            "Просто ответь на этот вопрос текстом или голосом — я на основе ответа "
            "исправлю и ДОПОЛНЮ состав приёма пищи."
        )
    else:
        lines.append(
            "\nЕсли хочешь что-то поправить или ДОБАВИТЬ (например, «там ещё сосиска» "
            "или «не было риса») — просто в следующем сообщении опиши это словами. "
            "Я пересчитаю именно этот приём пищи."
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_main_keyboard(),
    )


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        voice = update.message.voice or update.message.audio
        file = await voice.get_file()
        voice_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.exception("Ошибка загрузки аудио: %s", e)
        await update.message.reply_text("Не получилось скачать голосовое 😔 Попробуй ещё раз.")
        return

    text = await transcribe_voice(bytes(voice_bytes))
    if not text:
        await update.message.reply_text(
            "Не смогла распознать голос 😔 Попробуй сказать чуть чётче или короче."
        )
        return

    await text_message(update, context, forced_text=text)


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE, forced_text: str = None):
    user_id = update.effective_user.id
    text = forced_text if forced_text is not None else (update.message.text or "").strip()
    if not text:
        return

    state = user_states.setdefault(user_id, {})
    step = state.get("step")

    # кнопки
    if text == "📊 Саммари за день":
        await summary(update, context)
        return
    if text == "📜 История за день":
        await day_history_command(update, context)
        return
    if text == "📆 Как я ем неделю":
        await week_history_command(update, context)
        return
    if text == "❓ Помощь":
        await help_command(update, context)
        return

    # исправления после фото имеют приоритет
    if state.get("awaiting_correction") and state.get("pending_meal"):
        await handle_correction(update, context, text)
        return

    if is_correction_command(text):
        await handle_correction(update, context, text)
        return

    profile = user_profiles.get(user_id)

    # ===== АНКЕТА =====
    if step is None and profile is None:
        state["step"] = "ask_name"
        await update.message.reply_text(
            "Давай сначала познакомимся: как тебя зовут?",
            reply_markup=get_main_keyboard(),
        )
        return

    if step == "ask_name":
        name = text.strip()
        user_profiles[user_id] = {"name": name}
        save_profiles()
        state["step"] = "ask_age"
        await update.message.reply_text(
            f"Рада знакомству, {name}! 🎉 Сколько тебе полных лет?",
            reply_markup=get_main_keyboard(),
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
            "Укажи, пожалуйста, пол (м/ж). Это нужно для формулы BMR."
        )
        return

    if step == "ask_sex":
        sex = text.strip().lower()
        if not (sex.startswith("м") or sex.startswith("ж")):
            await update.message.reply_text("Напиши, пожалуйста, просто 'м' или 'ж' 🙂")
            return
        user_profiles[user_id]["sex"] = sex
        save_profiles()
        state["step"] = "ask_weight"
        await update.message.reply_text(
            "Теперь напиши свой текущий вес в килограммах (например, 70.5)."
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
            "Отлично! Теперь напиши рост в сантиметрах (например, 170)."
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
        state["step"] = "ask_activity"
        await update.message.reply_text(
            "Теперь давай оценим уровень твоей повседневной активности.\n"
            "Напиши цифру от 1 до 5:\n"
            "1 — минимальная активность (сидячая работа, почти без спорта)\n"
            "2 — лёгкая активность (1–3 нетяжёлые тренировки в неделю)\n"
            "3 — умеренная активность (3–5 тренировок в неделю)\n"
            "4 — высокая активность (6–7 тренировок в неделю)\n"
            "5 — очень высокая активность (физический труд или спорт почти каждый день)",
            reply_markup=get_main_keyboard(),
        )
        return

    if step == "ask_activity":
        parsed = parse_activity_level(text)
        if not parsed:
            await update.message.reply_text(
                "Пожалуйста, напиши число от 1 до 5, которое лучше всего описывает твою активность."
            )
            return
        activity_desc, activity_factor = parsed
        user_profiles[user_id]["activity_description"] = activity_desc
        user_profiles[user_id]["activity_factor"] = activity_factor
        save_profiles()
        state["step"] = "ask_target_weight"
        await update.message.reply_text(
            "До какого веса ты хочешь прийти? Напиши число в килограммах (например, 60)."
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
        state["step"] = "ask_health"
        await update.message.reply_text(
            "Есть ли у тебя какие-то особенности здоровья? (например: диабет, гастрит, инсулинорезистентность)\n"
            "Если нет, просто напиши 'нет'."
        )
        return

    if step == "ask_health":
        user_profiles[user_id]["health_conditions"] = text.strip()
        save_profiles()
        state["step"] = "ask_allergies"
        await update.message.reply_text(
            "Есть ли у тебя пищевые аллергии или непереносимости? (например: лактоза, глютен, орехи)\n"
            "Если нет, напиши 'нет'."
        )
        return

    if step == "ask_allergies":
        user_profiles[user_id]["allergies"] = text.strip()
        save_profiles()
        state["step"] = "ask_lifestyle"
        await update.message.reply_text(
            "Как у тебя со сном и уровнем стресса? (например: сплю 6ч, много стресса)\n"
            "Это поможет мне давать более точные советы."
        )
        return

    if step == "ask_lifestyle":
        user_profiles[user_id]["lifestyle"] = text.strip()
        save_profiles()
        state["step"] = "ask_months_to_goal"
        await update.message.reply_text(
            "За сколько МЕСЯЦЕВ ты бы хотела/хотел прийти к этому весу? Например: 6."
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

        bmr = calculate_bmr(
            weight=profile["weight"],
            height=profile["height"],
            age=profile["age"],
            sex=profile["sex"],
        )
        activity_factor = profile.get("activity_factor", 1.3)
        plan = plan_calories_and_deficit(
            bmr=bmr,
            current_weight=profile["weight"],
            target_weight=profile["target_weight"],
            months_to_goal=months,
            activity_factor=activity_factor,
        )
        goal_type = plan["goal_type"]
        maintenance = plan["maintenance_calories"]
        target_cal = plan["target_calories"]
        daily_def = plan["daily_deficit"]
        realistic_months = plan["realistic_months"]
        plan_comment = plan["plan_comment"]

        macros = calculate_macro_targets(
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
        profile.update(macros)
        save_profiles()

        state["step"] = None
        state.setdefault("active_day", today_str())

        goal_text = {
            "loss": "похудение",
            "gain": "набор веса",
            "maintain": "удержание веса",
        }[goal_type]

        tp = macros["target_protein"]
        tf = macros["target_fat"]
        tc = macros["target_carbs"]
        activity_desc = profile.get("activity_description", "")
        health = profile.get("health_conditions", "нет")
        allergies = profile.get("allergies", "нет")

        deficit_line = ""
        if goal_type == "loss":
            deficit_line = f"Ориентировочный дефицит: ~{abs(round(daily_def))} ккал/день."
        elif goal_type == "gain":
            deficit_line = f"Ориентировочный профицит: ~{abs(round(daily_def))} ккал/день."

        text_out = (
            "🎯 Твой профессиональный план готов!\n\n"
            "🧮 *Расчеты (Миффлин–Сан Жеор):*\n"
            f"• Базовый метаболизм (BMR): *{round(bmr)} ккал*\n"
            f"• Расход с активностью: *{round(maintenance)} ккал*\n\n"
            "📊 *Твои данные и цель:*\n"
            f"• Вес: {profile['weight']} ➔ {profile['target_weight']} кг\n"
            f"• Цель: {goal_text}\n"
            f"• Здоровье: {health}\n"
            f"• Аллергии: {allergies}\n\n"
            "🍎 *Целевое БЖУ на день:*\n"
            f"• Калории: *~{round(target_cal)} ккал*\n"
            f"• Белки: *~{round(tp)} г*\n"
            f"• Жиры: *~{round(tf)} г*\n"
            f"• Углеводы: *~{round(tc)} г*\n\n"
            f"💡 {deficit_line}\n"
            f"{plan_comment}\n\n"
            "🚀 Теперь просто присылай фото еды, этикеток или пиши /water [мл].\n"
            "Я буду следить за твоим прогрессом как настоящий диетолог!"
        )

        await update.message.reply_text(
            text_out,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_main_keyboard(),
        )
        return

    # если мы ещё в анкете, но сообщение не попало в шаг — не включаем коуча
    if step is not None:
        await update.message.reply_text(
            "Давай сначала закончим анкету — ответь, пожалуйста, на предыдущий вопрос 🙂"
        )
        return

    # ===== ОБЫЧНЫЙ ТЕКСТ: ЧАТ-КОУЧ (ТОЛЬКО ПОСЛЕ АНКЕТЫ) =====
    try:
        reply = await chat_with_coach(user_id, text)
        await update.message.reply_text(
            reply,
            reply_markup=get_main_keyboard(),
        )
    except Exception as e:
        logger.exception("Ошибка в чат-коуче: %s", e)
        await update.message.reply_text(f"Ошибка OpenAI в чате: {e}")


# =============== MAIN ===============

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("week", week_history_command))
    app.add_handler(CommandHandler("dayhistory", day_history_command))
    app.add_handler(CommandHandler("water", water_command))

    app.add_handler(MessageHandler(filters.PHOTO, photo_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

    logger.info("Бот запущен (polling). Ждём сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()
