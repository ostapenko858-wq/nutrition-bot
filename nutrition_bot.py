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
    """Формула Миффлина–Сан Жеора."""
    sex = sex.lower()
    if sex.startswith("ж"):
        s = -161
    else:
        s = 5
    return 10 * weight + 6.25 * height - 5 * age + s


def calculate_target_calories(bmr: float, goal: str) -> float:
    goal = goal.lower()
    if "похуд" in goal:
        return max(bmr - 300, 1200)
    elif "набор" in goal or "масса" in goal:
        return bmr + 300
    else:
        return bmr


def calculate_macro_targets(weight: float, target_calories: float, goal: str) -> Dict[str, float]:
    """
    Простые рекомендации по КБЖУ в зависимости от цели.
    Вес — в кг, калории — целевой колораж.
    Возвращает граммы: белки, жиры, углеводы.
    """
    g = goal.lower()
    if "похуд" in g:
        protein_g = 1.8 * weight
        fat_g = 0.8 * weight
    elif "поддерж" in g:
        protein_g = 1.6 * weight
        fat_g = 1.0 * weight
    else:  # набор массы или другое
        protein_g = 2.0 * weight
        fat_g = 1.0 * weight

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
    """Определяем, что пользователь хочет пересчитать/уточнить еду."""
    lowered = text.strip().lower()
    return (
        "исправ" in lowered
        or "уточн" in lowered
        or "пересчит" in lowered
    )


# ========= OPENAI: АНАЛИЗ ФОТО ЕДЫ / ЭТИКЕТКИ =========

async def analyze_food_image(image_bytes: bytes):
    """
    Анализирует фото еды ИЛИ фото этикетки через GPT-4o-mini (vision) и возвращает:
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
   - фото тарелки с едой / напитком
   - или фото этикетки/упаковки с таблицей КБЖУ.
2. Если это ТАРЕЛКА / ЕДА:
   - Определи основные блюда и напитки.
   - Для каждого оцени примерный размер порции в граммах и калорийность.
   - Также оцени белки, жиры, углеводы для каждого компонента.
   - Ориентируйся на типичные значения:
     - куриная грудка без кожи ~165 ккал/100 г, ~31 г белка
     - варёный рис ~130 ккал/100 г, ~2.5 г белка, ~0.3 г жира, ~28 г углеводов
     - макароны отварные ~150 ккал/100 г, ~5 г белка, ~1.5 г жира, ~30 г углеводов
     - жареный картофель ~200–230 ккал/100 г, ~3 г белка, ~8–10 г жира, ~30 г углеводов
     - салат из свежих овощей без масла ~20 ккал/100 г, ~1 г белка, ~0 г жира, ~4 г углеводов
     - растительное масло ~900 ккал/100 г, 0 г белка, 100 г жира, 0 г углеводов (10 г ~90 ккал)
     - сладкие напитки ~40–45 ккал на 100 мл, почти без белка и жира, углеводы ~10–11 г/100 мл
   - Порции должны быть реалистичными.
3. Если это ЭТИКЕТКА:
   - Прочитай таблицу КБЖУ.
   - Определи калорийность, белки, жиры, углеводы хотя бы:
     • на 100 г/мл
     • и, если указано, на порцию (например, 30 г, 1 батончик, 250 мл).
   - Если понятно, какая порция показана или предполагается (например, 1 батончик),
     оцени КБЖУ для ОДНОЙ порции как будто пользователь её съел.
   - Если размер порции неочевиден — предположи реалистичную порцию
     и при необходимости включи need_clarification=true с вопросом.
4. Посчитай суммарную калорийность, белки, жиры, углеводы по всему тому,
   что пользователь, скорее всего, съел/выпил.
5. Коротко опиши, что ты видишь на фото (один абзац).

Очень важно:
— Ты ДОЛЖЕН вернуть реалистичные значения калорий и КБЖУ.
— Старайся не завышать и не занижать в разы.
— Если очень неуверен в порции — попроси уточнение (need_clarification=true).

Ответ ДОЛЖЕН быть строго в формате JSON (без комментариев, текста до или после):

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

Если на фото совсем нет еды и нет этикетки с питательной ценностью, тогда верни:

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
        model="gpt-4o-mini",
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
        max_tokens=700,
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

Твоя задача:
1. Разбить описание на отдельные блюда/компоненты.
2. Для каждого оценить массу в граммах (если явно не указана — логично предположить)
   и калорийность, а также белки, жиры, углеводы.
   Ориентируйся на типичные значения:
   - куриная грудка без кожи ~165 ккал/100 г, ~31 г белка
   - варёный рис ~130 ккал/100 г, ~2.5 г белка, ~0.3 г жира, ~28 г углеводов
   - макароны отварные ~150 ккал/100 г, ~5 г белка, ~1.5 г жира, ~30 г углеводов
   - жареный картофель ~200–230 ккал/100 г, ~3 г белка, ~8–10 г жира, ~30 г углеводов
   - салат из свежих овощей без масла ~20 ккал/100 г
   - растительное масло ~900 ккал/100 г, 0 г белка, 100 г жира, 0 г углеводов
   - сладкие напитки ~40–45 ккал на 100 мл
3. Посчитать суммарную калорийность и КБЖУ.
4. Кратко описать приём пищи.

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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_description},
        ],
        temperature=0.25,
        max_tokens=600,
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

    # вырезаем часть до ключевого слова (исправ, уточн, пересчит)
    clean_text = original
    for kw in ["исправ", "уточн", "пересчит"]:
        idx = lowered.find(kw)
        if idx != -1:
            clean_text = original[idx + len(kw):].strip()
            break

    # если ещё есть двоеточие — вырезаем всё до него
    if ":" in clean_text:
        _, _, clean_text = clean_text.partition(":")
        clean_text = clean_text.strip()

    if not clean_text:
        await update.message.reply_text(
            "Опиши, пожалуйста, еду после слов 'исправление' или 'уточнение' — например:\n"
            "исправление: там была гречка 150 г, говядина 120 г и салат из огурцов."
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


# ========= ИСТОРИЯ ЗА ДЕНЬ / НЕДЕЛЮ =========

async def send_day_history(update: Update, user_id: int, day_str: str):
    """Отправить историю за конкретный день (с КБЖУ)."""
    if user_id not in user_profiles:
        await update.message.reply_text(
            "Мы ещё не знакомы 🙂 Нажми /start, чтобы заполнить анкету."
        )
        return

    entries = user_food_log[user_id].get(day_str, [])
    profile = user_profiles[user_id]
    target_cal = profile.get("target_calories")
    target_p = profile.get("target_protein")
    target_f = profile.get("target_fat")
    target_c = profile.get("target_carbs")

    if not entries:
        await update.message.reply_text(
            f"За {day_str} у тебя нет записей о еде 🌿"
        )
        return

    total_cal = sum(e.get("calories", 0) for e in entries)
    total_p = sum(e.get("protein", 0) for e in entries)
    total_f = sum(e.get("fat", 0) for e in entries)
    total_c = sum(e.get("carbs", 0) for e in entries)

    text_lines = [f"История за день {day_str}:\n"]
    for e in entries:
        desc = e["description"]
        cal = e.get("calories", 0)
        p = e.get("protein", 0)
        f = e.get("fat", 0)
        c = e.get("carbs", 0)
        text_lines.append(
            f"• {desc} — ~{round(cal)} ккал, Б {round(p)} г, Ж {round(f)} г, У {round(c)} г"
        )

    text_lines.append("\n-------------------")
    text_lines.append(f"📊 Всего за день: *{round(total_cal)} ккал*")
    text_lines.append(
        f"🥩 Белки: *{round(total_p)} г*, 🧈 Жиры: *{round(total_f)} г*, 🥔 Углеводы: *{round(total_c)} г*"
    )

    if target_cal:
        text_lines.append(
            f"\n🎯 Цель по калориям: ~{round(target_cal)} ккал/день"
        )
    if target_p and target_f and target_c:
        text_lines.append(
            "\nЦели по КБЖУ:\n"
            f"🥩 Белки: ~{round(target_p)} г/день\n"
            f"🧈 Жиры: ~{round(target_f)} г/день\n"
            f"🥔 Углеводы: ~{round(target_c)} г/день"
        )

    await update.message.reply_text(
        "\n".join(text_lines),
        parse_mode=ParseMode.MARKDOWN,
    )


async def week_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сводка за последнюю календарную неделю (7 дней)."""
    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text(
            "Мы ещё не знакомы 🙂 Нажми /start, чтобы заполнить анкету."
        )
        return

    profile = user_profiles[user_id]
    target_cal = profile.get("target_calories")
    target_p = profile.get("target_protein")
    target_f = profile.get("target_fat")
    target_c = profile.get("target_carbs")

    today = date.today()
    days_data = []  # (day_str, total_cal, total_p, total_f, total_c)

    for i in range(7):
        d = today - timedelta(days=i)
        day_str = d.isoformat()
        entries = user_food_log[user_id].get(day_str, [])
        if not entries:
            continue
        total_cal = sum(e.get("calories", 0) for e in entries)
        total_p = sum(e.get("protein", 0) for e in entries)
        total_f = sum(e.get("fat", 0) for e in entries)
        total_c = sum(e.get("carbs", 0) for e in entries)
        days_data.append((day_str, total_cal, total_p, total_f, total_c))

    if not days_data:
        await update.message.reply_text(
            "За последние 7 дней у тебя нет записей о еде 🌿"
        )
        return

    days_data.sort(key=lambda x: x[0])  # по дате

    lines = ["Как ты ел(а) за последнюю неделю:\n"]
    week_cal = week_p = week_f = week_c = 0.0

    for day_str, cal, p, f, c in days_data:
        lines.append(
            f"{day_str}: ~{round(cal)} ккал, Б {round(p)} г, Ж {round(f)} г, У {round(c)} г"
        )
        week_cal += cal
        week_p += p
        week_f += f
        week_c += c

    days_count = len(days_data)
    avg_cal = week_cal / days_count
    avg_p = week_p / days_count
    avg_f = week_f / days_count
    avg_c = week_c / days_count

    lines.append("\n-------------------")
    lines.append(
        f"📊 Итого за неделю (по дням с записями):\n"
        f"Калории: ~{round(week_cal)} ккал\n"
        f"Белки: ~{round(week_p)} г\n"
        f"Жиры: ~{round(week_f)} г\n"
        f"Углеводы: ~{round(week_c)} г"
    )

    lines.append(
        f"\nСредние значения в день (по {days_count} дн.):\n"
        f"~{round(avg_cal)} ккал, Б {round(avg_p)} г, Ж {round(avg_f)} г, У {round(avg_c)} г"
    )

    if target_cal:
        lines.append(
            f"\n🎯 Цель по калориям: ~{round(target_cal)} ккал/день"
        )
    if target_p and target_f and target_c:
        lines.append(
            "Цели по КБЖУ:\n"
            f"🥩 Белки: ~{round(target_p)} г/день\n"
            f"🧈 Жиры: ~{round(target_f)} г/день\n"
            f"🥔 Углеводы: ~{round(target_c)} г/день"
        )

    lines.append(
        "\nЕсли хочешь обсудить, что видно по твоему питанию за неделю, напиши:\n"
        "«Обсудить мой прогресс»."
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
    )


async def week_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await week_history(update, context)


async def day_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {"step": None})
    state["awaiting_history_day"] = True
    await update.message.reply_text(
        "За какой день показать историю? 📅\n"
        "Напиши: 'сегодня', 'вчера' или дату в формате ГГГГ-ММ-ДД (например, 2025-12-31)."
    )


# ========= ОСНОВНАЯ ЛОГИКА ТЕКСТА =========

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {"step": None})
    step = state.get("step")
    lowered = text.strip().lower()

    # ответ на вопрос "новый день?"
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

    # ответ на запрос "история за день" (ожидаем дату)
    if state.get("awaiting_history_day"):
        if user_id not in user_profiles:
            await update.message.reply_text(
                "Мы ещё не знакомы 🙂 Нажми /start, чтобы заполнить анкету."
            )
            state["awaiting_history_day"] = False
            return

        txt = lowered.strip()
        if txt.startswith("сегодня"):
            day_str = today_str()
        elif txt.startswith("вчера"):
            d = date.today() - timedelta(days=1)
            day_str = d.isoformat()
        else:
            try:
                day_str = date.fromisoformat(text.strip()).isoformat()
            except ValueError:
                await update.message.reply_text(
                    "Не поняла дату 🤔\n"
                    "Напиши 'сегодня', 'вчера' или дату в формате ГГГГ-ММ-ДД, например: 2025-12-31."
                )
                return

        state["awaiting_history_day"] = False
        await send_day_history(update, user_id, day_str)
        return

    if step is None and user_id not in user_profiles:
        await update.message.reply_text(
            "Я пока с тобой не знаком 😅\n"
            "Нажми /start, чтобы пройти короткую анкету."
        )
        return

    # исправление/уточнение последней фотки (в т.ч. голосом)
    if is_correction_command(text):
        await handle_correction(update, context, text)
        return

    # ===== анкета =====
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
            "Отлично!\nТеперь напиши свой вес в килограммах (например, 72.5)."
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
        state["step"] = "ask_goal"

        goal_keyboard = ReplyKeyboardMarkup(
            [
                [KeyboardButton("Похудение"), KeyboardButton("Поддержание")],
                [KeyboardButton("Набор мышечной массы")],
            ],
            resize_keyboard=True,
            one_time_keyboard=True,
        )

        await update.message.reply_text(
            "Какая у тебя цель?\n\n"
            "• Похудение\n"
            "• Поддержание\n"
            "• Набор мышечной массы",
            reply_markup=goal_keyboard,
        )
        return

    if step == "ask_goal":
        goal = text.strip().lower()
        if not (
            "похуд" in goal
            or "поддерж" in goal
            or "набор" in goal
            or "масса" in goal
        ):
            await update.message.reply_text(
                "Выбери, пожалуйста, цель с клавиатуры или напиши что-то вроде "
                "'Похудение', 'Поддержание', 'Набор мышечной массы'."
            )
            return

        profile = user_profiles[user_id]
        profile["goal"] = text.strip()

        bmr = calculate_bmr(
            weight=profile["weight"],
            height=profile["height"],
            age=profile["age"],
            sex=profile["sex"],
        )
        target_cal = calculate_target_calories(bmr, goal)
        profile["bmr"] = bmr
        profile["target_calories"] = target_cal

        macro_targets = calculate_macro_targets(
            weight=profile["weight"],
            target_calories=target_cal,
            goal=goal,
        )
        profile.update(macro_targets)
        save_profiles()

        state["step"] = None
        state.setdefault("active_day", today_str())

        tp = macro_targets["target_protein"]
        tf = macro_targets["target_fat"]
        tc = macro_targets["target_carbs"]

        await update.message.reply_text(
            "Готово! Я посчитал твои базовые показатели 🔢\n\n"
            f"📊 Базовый метаболизм (BMR): *{round(bmr)} ккал/день*\n"
            f"🎯 Рекомендованный колораж: *~{round(target_cal)} ккал/день*\n\n"
            "Рекомендации по КБЖУ для твоей цели:\n"
            f"🥩 Белки: *~{round(tp)} г/день*\n"
            f"🧈 Жиры: *~{round(tf)} г/день*\n"
            f"🥔 Углеводы: *~{round(tc)} г/день*\n\n"
            "Теперь ты можешь отправлять мне *фото еды или этикеток*, а я буду примерно считать калории и КБЖУ.\n"
            "А ещё ты можешь писать (или говорить голосом) про свои переживания вокруг еды — обсудим без осуждения.\n\n"
            "‼️ Важно: в конце дня *добавляй к своим итогам 200–300 ккал* на возможную ошибку "
            "распознавания и взвешивания блюд.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # ===== пользователь уже зарегистрирован =====

    # кнопка: История за день
    if lowered.startswith("история за день"):
        state["awaiting_history_day"] = True
        await update.message.reply_text(
            "За какой день показать историю? 📅\n"
            "Напиши: 'сегодня', 'вчера' или дату в формате ГГГГ-ММ-ДД (например, 2025-12-31)."
        )
        return

    # кнопка: Как я ем неделю
    if lowered.startswith("как я ем неделю") or "как я ем неделю" in lowered:
        await week_history(update, context)
        return

    # кнопки прогресса
    if lowered.startswith("обсудить мой прогресс"):
        reply = await chat_with_coach(
            user_id,
            "Помоги мне обсудить мой сегодняшний прогресс по питанию, калориям и КБЖУ.",
        )
        await update.message.reply_text(reply)
        return

    if lowered.startswith("обсудить, почему недоел"):
        reply = await chat_with_coach(
            user_id,
            "Я сегодня недоел относительно своей цели по калориям. Помоги разобраться, почему так могло быть и что с этим делать.",
        )
        await update.message.reply_text(reply)
        return

    if lowered.startswith("обсудить переедание"):
        reply = await chat_with_coach(
            user_id,
            "Я сегодня вышел за пределы своего калоража и переел. Помоги разобраться, почему так получилось и как бережно к себе отнестись.",
        )
        await update.message.reply_text(reply)
        return

    if any(word in lowered for word in ("итоги дня", "итоги", "summary")):
        await summary(update, context)
        return

    # всё остальное — разговор с коучем
    try:
        reply = await chat_with_coach(user_id, text)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.exception("Ошибка в чат-коуче: %s", e)
        await update.message.reply_text(f"Ошибка OpenAI в чате: {e}")


# ========= START / RESET =========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id in user_profiles and "bmr" in user_profiles[user_id]:
        profile = user_profiles[user_id]
        name = profile.get("name", "друг")
        target_cal = profile.get("target_calories")
        target_p = profile.get("target_protein")
        target_f = profile.get("target_fat")
        target_c = profile.get("target_carbs")

        text = f"Снова ты, {name}! 💚\nЯ помню твои данные и цель.\n\n"
        if target_cal:
            text += f"🎯 Твой целевой колораж: ~{round(target_cal)} ккал/день.\n"
        if target_p and target_f and target_c:
            text += (
                f"🥩 Белки: ~{round(target_p)} г/день\n"
                f"🧈 Жиры: ~{round(target_f)} г/день\n"
                f"🥔 Углеводы: ~{round(target_c)} г/день\n\n"
            )
        text += (
            "Можешь сразу отправлять фото еды или этикетки, а также голос/текст "
            "про свои мысли и переживания вокруг еды.\n\n"
            "Если хочешь заново заполнить анкету — используй команду /reset."
        )
        state = user_states.setdefault(user_id, {})
        state.setdefault("step", None)
        state.setdefault("active_day", today_str())
        await update.message.reply_text(text)
        return

    user_states[user_id] = {"step": "ask_name"}

    await update.message.reply_text(
        "Привет! Я твой нутрициолог-бот 🤖🥦\n"
        "Помогу отслеживать питание и поговорить о твоих отношениях с едой.\n\n"
        "Давай познакомимся. Как тебя зовут?"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_profiles:
        del user_profiles[user_id]
        save_profiles()
    user_states[user_id] = {"step": "ask_name"}
    await update.message.reply_text(
        "Окей, давай начнём сначала 🧾\nКак тебя зовут?"
    )


# ========= ОБРАБОТЧИКИ TELEGRAM =========

async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    await handle_text(update, context, text)


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice or update.message.audio
    if voice is None:
        return

    file = await voice.get_file()
    voice_bytes = await file.download_as_bytearray()

    text = await transcribe_voice(voice_bytes)

    if not text:
        await update.message.reply_text(
            "Пока не получилось распознать голос 😔\n"
            "Отправь, пожалуйста, это сообщение текстом."
        )
        return

    await handle_text(update, context, text)


async def photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text(
            "Сначала давай познакомимся 🙂\nНажми /start и заполни анкету."
        )
        return

    state = user_states.setdefault(user_id, {"step": None})
    active_day = state.get("active_day") or today_str()
    state["active_day"] = active_day

    photo_list = update.message.photo
    if not photo_list:
        return

    photo = photo_list[-1]
    file = await photo.get_file()
    image_bytes = await file.download_as_bytearray()

    try:
        (
            description,
            items,
            total_cal,
            total_p,
            total_f,
            total_c,
            need_clarification,
            clarification_q,
        ) = await analyze_food_image(image_bytes)
    except Exception as e:
        logger.exception("Ошибка при анализе фото: %s", e)
        await update.message.reply_text(
            "Что-то пошло не так при анализе фото 😔\n"
            f"Ошибка OpenAI: {e}"
        )
        return

    day = active_day
    before_len = len(user_food_log[user_id][day])

    for item in items:
        user_food_log[user_id][day].append(
            {
                "description": item.get("name", "Блюдо"),
                "calories": float(item.get("calories", 0) or 0),
                "protein": float(item.get("protein", 0) or 0),
                "fat": float(item.get("fat", 0) or 0),
                "carbs": float(item.get("carbs", 0) or 0),
            }
        )

    added_count = len(user_food_log[user_id][day]) - before_len

    # запоминаем, что именно добавили по этой фотке, для исправлений/уточнений
    state["pending_meal"] = {"day": day, "count": added_count}

    lines = []
    if description:
        lines.append(f"📷 {description}\n")
    else:
        lines.append("Я посмотрел на фото и вот что понял:\n")

    if items:
        lines.append("Я думаю, на фото примерно:")
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
        if description:
            lines.append(f"Я вижу: {description}")
        lines.append(
            "Я не нашёл на этом фото еды или понятной этикетки, поэтому калории сюда не засчитываю.\n"
        )

    if need_clarification and clarification_q:
        lines.append(
            f"\n💭 Чтобы посчитать точнее, мне нужно уточнение:\n"
            f"_{clarification_q}_\n\n"
            "Можешь ответить голосом или текстом в формате:\n"
            "'уточнение: куриная грудка ~150 г, рис ~200 г без масла'."
        )
    else:
        lines.append(
            "\nЕсли я что-то перепутал или хочешь точнее по граммам и КБЖУ, напиши, пожалуйста, сообщение вида:\n"
            "_исправление: там была говядина 150 г и картофель 200 г без соуса_ — "
            "я пересчитаю калории и обновлю запись."
        )

    lines.append(
        "\n⚠️ Это только оценка. В конце дня добавь *200–300 ккал* к общему итогу "
        "на возможную погрешность."
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
    )


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if user_id not in user_profiles:
        await update.message.reply_text(
            "Мы ещё не знакомы 🙂 Нажми /start, чтобы заполнить анкету."
        )
        return

    state = user_states.setdefault(user_id, {"step": None})
    active_day = state.get("active_day") or today_str()
    state["active_day"] = active_day

    entries = user_food_log[user_id].get(active_day, [])

    profile = user_profiles[user_id]
    target_cal = profile.get("target_calories")
    target_p = profile.get("target_protein")
    target_f = profile.get("target_fat")
    target_c = profile.get("target_carbs")

    if not entries:
        await update.message.reply_text(
            "На этот день у тебя пока нет записей о еде.\n"
            "Отправь фото приёмов пищи или этикеток, и я начну считать калории и КБЖУ 📸"
        )
        return

    total_cal = sum(e.get("calories", 0) for e in entries)
    total_p = sum(e.get("protein", 0) for e in entries)
    total_f = sum(e.get("fat", 0) for e in entries)
    total_c = sum(e.get("carbs", 0) for e in entries)

    total_with_margin = total_cal + 250  # запас на погрешность

    lines = [f"Итоги за день ({active_day}):\n"]

    for e in entries:
        desc = e["description"]
        cal = e.get("calories", 0)
        p = e.get("protein", 0)
        f = e.get("fat", 0)
        c = e.get("carbs", 0)
        lines.append(
            f"• {desc} — ~{round(cal)} ккал, Б {round(p)} г, Ж {round(f)} г, У {round(c)} г"
        )

    lines.append("\n-------------------")
    lines.append(f"📊 Всего за день: *{round(total_cal)} ккал*")
    lines.append(
        f"➕ С учётом погрешности (+~250 ккал): *{round(total_with_margin)} ккал*"
    )
    lines.append(
        f"\n🥩 Белки за день: *{round(total_p)} г*\n"
        f"🧈 Жиры за день: *{round(total_f)} г*\n"
        f"🥔 Углеводы за день: *{round(total_c)} г*"
    )

    progress_comment = None
    rows: List[List[KeyboardButton]] = []

    if target_cal:
        diff = total_with_margin - target_cal
        abs_diff = abs(diff)
        sign = "выше" if diff > 0 else "ниже"
        lines.append(
            f"\n🎯 Твоя цель по калориям: ~{round(target_cal)} ккал/день\n"
            f"Разница с целью (с учётом погрешности): *{abs(round(diff))} ккал {sign}*."
        )

        margin = target_cal * 0.1
        if abs_diff <= margin:
            progress_comment = "О, сегодня ты круто справился — был примерно в своём диапазоне 💚"
            rows.append([KeyboardButton("Обсудить мой прогресс")])
        elif diff < 0:
            progress_comment = "Так, а чего это ты недоел сегодня? 🤔 Давай мягко разберёмся, что случилось."
            rows.append([KeyboardButton("Обсудить, почему недоел")])
        else:
            progress_comment = "Малыш, ты сегодня явно пере́ел 😅 Давай обсудим это без обвинений, ладно?"
            rows.append([KeyboardButton("Обсудить переедание")])

    if target_p and target_f and target_c:
        lines.append(
            "\nЦели по КБЖУ:\n"
            f"🥩 Белки: ~{round(target_p)} г/день\n"
            f"🧈 Жиры: ~{round(target_f)} г/день\n"
            f"🥔 Углеводы: ~{round(target_c)} г/день"
        )

    lines.append(
        "\nПомни, что это примерные оценки. Для серьёзных изменений в питании "
        "лучше дополнительно консультироваться с врачом или диетологом."
    )

    # добавляем наши новые кнопки в клавиатуру
    rows.append([KeyboardButton("История за день"), KeyboardButton("Как я ем неделю")])
    keyboard = ReplyKeyboardMarkup(rows, resize_keyboard=True)

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard,
    )

    if progress_comment:
        await update.message.reply_text(progress_comment)

    # Спрашиваем про новый день
    state["awaiting_new_day_answer"] = True
    yn_keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("Да, новый день"), KeyboardButton("Нет, продолжаем этот день")]],
        resize_keyboard=True,
    )
    await update.message.reply_text(
        "Скажи, у тебя уже начался новый день по ощущениям? 🌅\n"
        "Если да — я начну считать следующий приём пищи как новый день.\n"
        "Если нет — продолжим копить калории и КБЖУ в этом же дне.",
        reply_markup=yn_keyboard,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я нутрициолог-бот 🤖\n\n"
        "/start — начать или просто поздороваться, если мы уже знакомы\n"
        "/reset — сбросить анкету и заполнить заново\n"
        "/summary — итоги текущего дня\n"
        "/dayhistory — история по выбранному дню\n"
        "/week — как ты ел(а) за последнюю неделю\n\n"
        "Что я умею:\n"
        "• По фото еды считаю примерные калории и КБЖУ.\n"
        "• По фото ЭТИКЕТКИ считываю таблицу КБЖУ и считаю порцию.\n"
        "• По тексту или голосу могу пересчитать приём пищи.\n"
        "• Даю рекомендации по дневным КБЖУ с учётом твоей цели.\n"
        "• Могу поговорить с тобой про эмоции и психологию питания.\n\n"
        "Если я ошибся с фото или хочешь точнее по граммам, используй формат:\n"
        "исправление: куриная грудка 150 г и рис 200 г без масла\n"
        "или\n"
        "уточнение: гречка 120 г, говядина 100 г, салат из огурцов без масла."
    )


# ========= ЗАПУСК ПРИЛОЖЕНИЯ =========

def main():
    # Строим приложение (здесь всё как у тебя локально)
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
