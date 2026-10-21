import asyncio
import base64
import json
import logging
import os
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
import html
import io
import httpx
import urllib.parse
from card_generator import render_day_card
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =============== НАСТРОЙКИ И КЛИЕНТЫ ===============

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path.home() / ".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
TRAINER_CONTACT = os.getenv("TRAINER_CONTACT", "своему тренеру")
TRAINER_USERNAME = os.getenv("TRAINER_USERNAME", "VolkovaYasno").lstrip("@")

if not TELEGRAM_TOKEN:
    raise RuntimeError(
        "Не найден TELEGRAM_TOKEN в переменных окружения или файле .env."
    )
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Не найден GEMINI_API_KEY в переменных окружения или файле .env."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def _prepare_contents(input_data: Any) -> List[Any]:
    contents: List[Any] = []
    if isinstance(input_data, str):
        contents.append(input_data)
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                contents.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    contents.append(item.get("text", ""))
                elif item_type in ("image", "audio"):
                    data = item.get("data", "")
                    mime_type = item.get(
                        "mime_type",
                        "image/jpeg" if item_type == "image" else "audio/ogg",
                    )
                    if isinstance(data, str):
                        raw_bytes = base64.b64decode(data)
                    elif isinstance(data, bytes):
                        raw_bytes = data
                    else:
                        continue
                    contents.append(
                        types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)
                    )
                else:
                    contents.append(str(item))
            elif isinstance(item, types.Part):
                contents.append(item)
            else:
                contents.append(str(item))
    else:
        contents.append(str(input_data))
    return contents


async def create_interaction(
    model: str,
    input_data: Any,
    system_instruction: Optional[str] = None,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Вызывает Gemini API через официальный SDK google-genai с поддержкой:
    1) асинхронного клиента client.aio.models.generate_content
    2) структурированных JSON схем (response_schema)
    3) мультимодальных входов (текст, фото, голос)
    4) автоматического retry с экспоненциальной задержкой
    5) автоматического failover на стабильные резервные модели (gemini-2.5-flash, gemini-3.5-flash, gemini-flash-latest) при 503 / 429 / перегрузке.
    """
    candidate_models = [model, "gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]
    ordered_models: List[str] = []
    for m in candidate_models:
        if m and m not in ordered_models:
            ordered_models.append(m)

    contents = _prepare_contents(input_data)

    config_kwargs: Dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    last_error = None
    for current_model in ordered_models:
        for attempt in range(2):
            try:
                resp = await client.aio.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config,
                )
                text = (resp.text or "").strip()
                if text:
                    return text
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                is_transient = any(
                    code in err_msg
                    for code in [
                        "503",
                        "429",
                        "500",
                        "unavailable",
                        "resource_exhausted",
                        "deadline",
                        "timeout",
                    ]
                )
                logger.warning(
                    "Gemini (%s, попытка %d/2) ошибка: %s",
                    current_model,
                    attempt + 1,
                    e,
                )
                if attempt == 0 and is_transient:
                    await asyncio.sleep(1.0)
                    continue
                break

    try:
        payload: Dict[str, Any] = {
            "model": ordered_models[0] if ordered_models else "gemini-2.5-flash",
            "input": input_data,
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction
        if response_schema:
            payload["response_format"] = response_schema

        async with httpx.AsyncClient(timeout=45.0) as http:
            resp = await http.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body.get("output_text"), str):
                return body["output_text"].strip()
    except Exception as e:
        logger.error("REST interactions fallback also failed: %s", e)

    raise last_error or RuntimeError("Все модели Gemini не смогли вернуть ответ.")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =============== ПЕРСИСТЕНТНОЕ ХРАНИЛИЩЕ ===============

PROFILE_FILE = BASE_DIR / "profiles.json"
STORAGE_FILE = BASE_DIR / "bot_storage.json"

# user_states:
#   step: Optional[str] (анкета, чек-ин, ожидание текста еды)
#   active_day: str
#   active_day_updated_on: str
#   awaiting_correction: bool
#   awaiting_new_day_answer: bool
#   awaiting_history_day: bool
#   awaiting_food_text: bool
#   pending_meal: {"day": str, "count": int, "items": [...]}
user_states: Dict[int, Dict[str, Any]] = {}

# анкета и цели + КБЖУ
user_profiles: Dict[int, Dict[str, Any]] = {}

# user_food_log: user -> day -> [ {description, grams, calories, protein, fat, carbs}, ... ]
user_food_log: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(
    lambda: defaultdict(list)
)

# трекер воды: user -> day -> ml
user_water_log: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

# чек-ины состояния/сна/готовности: user -> day -> {sleep, mood, readiness, needs_trainer, ai_reply}
user_checkins: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)

# история общения с коучем
user_chat_history: Dict[int, List[Dict[str, str]]] = defaultdict(list)

# уровни активности
ACTIVITY_LEVELS: Dict[str, Tuple[str, float]] = {
    "1": ("минимальная активность (сидячая работа, почти без спорта)", 1.2),
    "2": ("лёгкая активность (1–3 нетяжёлые тренировки в неделю)", 1.375),
    "3": ("умеренная активность (3–5 тренировок в неделю)", 1.55),
    "4": ("высокая активность (6–7 тренировок в неделю)", 1.725),
    "5": ("очень высокая активность (физический труд или спорт почти каждый день)", 1.9),
}


# =============== УТИЛИТЫ И РАСЧЁТЫ ===============

def parse_float(text: str) -> float:
    return float(text.replace(",", ".").strip())


def today_str() -> str:
    return date.today().isoformat()


def get_active_day(user_id: int) -> str:
    """
    Возвращает активный день пользователя.
    Если наступил новый календарный день, автоматически переключает active_day на сегодня,
    чтобы записи не застревали во вчерашней дате.
    """
    state = user_states.setdefault(user_id, {})
    real_today = today_str()
    active_day = state.get("active_day")
    updated_on = state.get("active_day_updated_on")

    if not active_day or updated_on != real_today:
        state["active_day"] = real_today
        state["active_day_updated_on"] = real_today
        return real_today

    return active_day


def calculate_bmr(weight: float, height: float, age: int, sex: str) -> float:
    """
    Формула Миффлина–Сан Жеора:
    Мужчины: BMR = 10*вес + 6.25*рост - 5*возраст + 5
    Женщины: BMR = 10*вес + 6.25*рост - 5*возраст - 161
    """
    sex = sex.lower()
    s = -161 if sex.startswith("ж") else 5
    return 10 * weight + 6.25 * height - 5 * age + s


def infer_goal_type(current_weight: float, target_weight: float, goal_str: str = "") -> str:
    if current_weight and target_weight:
        diff = target_weight - current_weight
        if diff < -1:
            return "loss"
        if diff > 1:
            return "gain"
        return "maintain"
    g = (goal_str or "").lower()
    if "похуд" in g or "сниж" in g:
        return "loss"
    if "набор" in g or "масс" in g:
        return "gain"
    return "maintain"


def parse_activity_level(text: str) -> Optional[Tuple[str, float]]:
    t = text.strip().lower()
    for key in ACTIVITY_LEVELS:
        if t == key or t.startswith(f"{key} ") or t.startswith(f"{key}.") or t.startswith(f"{key}-"):
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
                f"понадобится примерно {realistic_months:.1f} мес. при "
                f"дефиците около {daily_deficit:.0f} ккал/день. "
                f"Это чуть дольше выбранных {months_to_goal:.1f} мес., "
                "зато бережно для здоровья и без срывов."
            )
        elif required_deficit < min_deficit:
            daily_deficit = min_deficit
            realistic_days = total_deficit / daily_deficit
            realistic_months = realistic_days / 30.0
            plan_comment = (
                f"Цель комфортная: при мягком дефиците ~{daily_deficit:.0f} ккал/день "
                f"переход с {current_weight:.1f} до {target_weight:.1f} кг займёт около "
                f"{realistic_months:.1f} мес."
            )
        else:
            daily_deficit = required_deficit
            realistic_months = months_to_goal
            plan_comment = (
                f"Чтобы прийти с {current_weight:.1f} до {target_weight:.1f} кг за "
                f"{months_to_goal:.1f} мес., нужен дефицит около "
                f"{daily_deficit:.0f} ккал/день — это в комфортных и безопасных пределах."
            )

        target_calories = max(maintenance - daily_deficit, 1200.0)

    elif goal_type == "gain":
        kg_to_gain = target_weight - current_weight
        total_surplus = kg_to_gain * 7700
        daily_deficit = -300.0
        realistic_days = total_surplus / abs(daily_deficit)
        realistic_months = realistic_days / 30.0
        target_calories = maintenance + abs(daily_deficit)
        plan_comment = (
            f"Для качественного набора с {current_weight:.1f} до {target_weight:.1f} кг "
            f"заложен мягкий профицит ~300 ккал/день (около {realistic_months:.1f} мес.)."
        )
    else:
        daily_deficit = 0.0
        target_calories = maintenance
        plan_comment = (
            "Цель — комфортное поддержание веса и формирование устойчивой привычки питания."
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
    base_w = target_weight if target_weight else current_weight
    if goal_type == "loss":
        protein_g = 1.8 * base_w
        fat_g = 0.8 * base_w
    elif goal_type == "gain":
        protein_g = 2.0 * base_w
        fat_g = 1.0 * base_w
    else:
        base_w = current_weight or base_w
        protein_g = 1.6 * base_w
        fat_g = 1.0 * base_w

    cal_pf = protein_g * 4 + fat_g * 9
    carbs_cal = max(target_calories - cal_pf, target_calories * 0.3)
    carbs_g = carbs_cal / 4.0

    return {
        "target_protein": round(protein_g, 1),
        "target_fat": round(fat_g, 1),
        "target_carbs": round(carbs_g, 1),
    }


def ensure_profile_macros(profile: Dict[str, Any]) -> None:
    """Досчитывает целевые БЖУ для старых профилей из profiles.json, если их нет."""
    if not profile or "target_calories" not in profile:
        return
    if all(k in profile for k in ("target_protein", "target_fat", "target_carbs")):
        return
    weight = float(profile.get("weight", 65.0))
    target_weight = float(profile.get("target_weight", weight))
    target_cal = float(profile.get("target_calories", 1500.0))
    goal_type = profile.get("goal_type") or infer_goal_type(
        weight, target_weight, str(profile.get("goal", ""))
    )
    macros = calculate_macro_targets(weight, target_weight, target_cal, goal_type)
    profile.update(macros)


# =============== ЗАГРУЗКА И СОХРАНЕНИЕ ДАННЫХ ===============

def _atomic_write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_all_data() -> None:
    global user_profiles
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            user_profiles = {int(uid): prof for uid, prof in data.items()}
            for prof in user_profiles.values():
                ensure_profile_macros(prof)
            logger.info("Загружено профилей пользователей: %d", len(user_profiles))
        except Exception as e:
            logger.exception("Ошибка при загрузке profiles.json: %s", e)
            user_profiles = {}

    if STORAGE_FILE.exists():
        try:
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for uid_str, days_map in raw.get("food_log", {}).items():
                uid = int(uid_str)
                for day_s, items in days_map.items():
                    user_food_log[uid][day_s] = items

            for uid_str, days_map in raw.get("water_log", {}).items():
                uid = int(uid_str)
                for day_s, ml in days_map.items():
                    user_water_log[uid][day_s] = int(ml)

            for uid_str, days_map in raw.get("checkins", {}).items():
                uid = int(uid_str)
                user_checkins[uid] = days_map

            for uid_str, hist in raw.get("chat_history", {}).items():
                user_chat_history[int(uid_str)] = hist

            for uid_str, st in raw.get("user_states", {}).items():
                user_states[int(uid_str)] = st

            logger.info("Загружено хранилище bot_storage.json")
        except Exception as e:
            logger.exception("Ошибка при загрузке bot_storage.json: %s", e)


def save_profiles() -> None:
    try:
        data = {str(uid): profile for uid, profile in user_profiles.items()}
        _atomic_write_json(PROFILE_FILE, data)
    except Exception as e:
        logger.exception("Ошибка при сохранении profiles.json: %s", e)


def save_storage() -> None:
    try:
        payload = {
            "food_log": {str(uid): dict(days) for uid, days in user_food_log.items()},
            "water_log": {str(uid): dict(days) for uid, days in user_water_log.items()},
            "checkins": {str(uid): dict(days) for uid, days in user_checkins.items()},
            "chat_history": {str(uid): hist[-20:] for uid, hist in user_chat_history.items()},
            "user_states": {str(uid): st for uid, st in user_states.items()},
        }
        _atomic_write_json(STORAGE_FILE, payload)
    except Exception as e:
        logger.exception("Ошибка при сохранении bot_storage.json: %s", e)


load_all_data()
save_profiles()


# =============== КЛАВИАТУРЫ И БЕЗОПАСНАЯ ОТПРАВКА ===============

def get_trainer_url(message_text: str = "Привет, у меня сегодня не очень ощущения, нужна с тобой беседа") -> str:
    username = (os.getenv("TRAINER_USERNAME") or "VolkovaYasno").lstrip("@")
    encoded = urllib.parse.quote(message_text)
    return f"https://t.me/{username}?text={encoded}"


def get_trainer_inline_keyboard(
    button_text: str = "💬 Написать тренеру",
    message_text: str = "Привет, у меня сегодня не очень ощущения, нужна с тобой беседа",
) -> InlineKeyboardMarkup:
    url = get_trainer_url(message_text)
    return InlineKeyboardMarkup([[InlineKeyboardButton(button_text, url=url)]])


def render_progress_bar(current: float, target: float, length: int = 10) -> str:
    """Генерирует аккуратный текстовый прогресс-бар калорий или воды."""
    if not target or target <= 0:
        return ""
    pct = max(current / target, 0.0)
    filled = int(round(min(pct, 1.0) * length))
    empty = max(length - filled, 0)
    bar = "▓" * filled + "░" * empty
    return f"`[{bar}]` *{round(pct * 100)}%*"


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Компактное, чистое главное меню в 2 ряда."""
    buttons = [
        [
            KeyboardButton("☀️ Настрой"),
            KeyboardButton("🍽 Добавить еду"),
            KeyboardButton("💧 +Вода"),
        ],
        [
            KeyboardButton("📊 Итог дня"),
            KeyboardButton("📆 Неделя"),
            KeyboardButton("💬 Тренер"),
        ],
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True)


async def safe_reply(
    update: Update,
    text: str,
    reply_markup: Optional[Any] = None,
    parse_mode: Optional[str] = ParseMode.MARKDOWN,
) -> None:
    """Отправляет сообщение с фоллбэком на обычный текст, если разметка не распарсилась."""
    message = update.message or (update.callback_query.message if update.callback_query else None)
    if not message:
        return
    try:
        await message.reply_text(
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
    except BadRequest:
        clean_text = re.sub(r"<[^>]+>", "", text).replace("*", "").replace("_", "").replace("`", "")
        await message.reply_text(
            clean_text,
            reply_markup=reply_markup,
        )


async def safe_reply_photo(
    update: Update,
    photo: io.BytesIO,
    caption: str = "",
    reply_markup: Optional[Any] = None,
    parse_mode: Optional[str] = ParseMode.HTML,
) -> None:
    """Отправляет фото-карточку с подписью или падает обратно на текст при ошибке."""
    message = update.message or (update.callback_query.message if update.callback_query else None)
    if not message:
        return
    try:
        photo.seek(0)
        await message.reply_photo(
            photo=photo,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
    except Exception as e:
        logger.warning("Не удалось отправить фото-карточку: %s. Отправляю текстом.", e)
        await safe_reply(update, caption, reply_markup=reply_markup, parse_mode=parse_mode)


# =============== РАСПОЗНАВАНИЕ КОМАНД И ТЕКСТА ЕДЫ ===============

CORRECTION_PREFIX_RE = re.compile(
    r"^\s*(?:исправ(?:ь|ить|ление)?|уточн(?:и|ить|ение)?|скоррект(?:ируй|ировать)?|"
    r"поправ(?:ь|ить|ка)?|пересчит(?:ай|ать)?|не\s+то|не\s+так|ошиб(?:ка|ся|лась)?)\s*[:,-]?\s*",
    re.IGNORECASE,
)


def is_correction_command(text: str) -> bool:
    """Проверяет, начинается ли сообщение со слов-исправлений."""
    return bool(CORRECTION_PREFIX_RE.match(text.strip()))


def strip_correction_prefix(text: str) -> str:
    """Аккуратно убирает только префикс в начале строки, не обрезая середину фразы."""
    cleaned = CORRECTION_PREFIX_RE.sub("", text.strip(), count=1).strip()
    return cleaned


def is_likely_food_entry(text: str) -> bool:
    """
    Определяет, похоже ли сообщение пользователя на запись еды, а не на вопрос или эмоциональную реплику коучу.
    """
    lowered = text.strip().lower()

    # Вопросы коучу без конкретного указания граммовок/порций отправляем в чат с коучем
    if "?" in lowered and not re.search(r"\b\d+\s*(?:г|гр|грамм|мл|л|шт|штук|кусоч|ложк|яиц|яйц)", lowered):
        return False

    # Чисто эмоциональные маркеры (проверяем по целым словам или шаблонам, чтобы не ловить 'половина' или 'свинина' на 'вина')
    purely_emotional = (
        r"\bчувств\w*\s+вин\w*",
        r"\bиспытыва\w*\s+вин\w*",
        r"\bмне\s+стыдно\b",
        r"\bстыдно\b",
        r"\bстресс\w*\b",
        r"\bтревог\w*\b",
        r"\bпочему я\b",
        r"\bпомоги\b",
        r"\bне могу сдержат\w*",
        r"\bсорвал\w*сь\b",
        r"\bсорвал\w*ся\b",
    )

    has_meal_prefix = any(
        p in lowered
        for p in (
            "на завтрак",
            "на обед",
            "на ужин",
            "на полдник",
            "на перекус",
            "мой завтрак",
            "мой обед",
            "мой ужин",
            "съел",
            "съела",
            "поел",
            "поела",
            "выпил",
            "выпила",
            "запиши",
            "завтрак:",
            "обед:",
            "ужин:",
            "перекус:",
        )
    )

    has_units = bool(
        re.search(
            r"\b\d+\s*(?:г|гр|грамм\w*|мл|л|литр\w*|шт|штук\w*|кусоч\w*|кусок\w*|ложк\w*|яиц\w*|яйц\w*|ккал)\b",
            lowered,
        )
    )

    food_keywords = (
        "шакшук", "яичниц", "омлет", "овсянк", "каш", "творог", "сырник",
        "бутерброд", "тост", "хлеб", "сыр", "куриц", "индейк", "говядин",
        "свинин", "рыб", "гречк", "рис", "макарон", "паст", "картофел",
        "пюре", "салат", "суп", "борщ", "кофе", "чай", "молок", "сливк",
        "йогурт", "кефир", "банан", "яблок", "ягод", "пицц", "бургер"
    )
    has_food_keyword = any(k in lowered for k in food_keywords)

    if any(re.search(pat, lowered) for pat in purely_emotional) and not (has_meal_prefix or has_units):
        return False

    return has_meal_prefix or has_units or has_food_keyword


def format_profile_context(user_id: int) -> str:
    """Формирует краткую карточку пользователя для системных промптов Gemini."""
    profile = user_profiles.get(user_id)
    if not profile:
        return "Профиль пользователя ещё не заполнен."

    day = get_active_day(user_id)
    total_cal, total_p, total_f, total_c, entries = aggregate_day(user_id, day)
    water = user_water_log[user_id].get(day, 0)
    checkin = user_checkins.get(user_id, {}).get(day)

    lines = [
        f"Имя: {profile.get('name', 'Пользователь')}",
        f"Возраст: {profile.get('age', '—')}, Пол: {profile.get('sex', '—')}",
        f"Текущий вес: {profile.get('weight', '—')} кг, Рост: {profile.get('height', '—')} см",
        f"Целевой вес: {profile.get('target_weight', profile.get('weight', '—'))} кг",
        f"Цель: {profile.get('goal') or profile.get('goal_type', 'поддержание')}",
        f"Целевой калораж: ~{round(profile.get('target_calories', 0))} ккал/день "
        f"(Б ~{round(profile.get('target_protein', 0))} г, "
        f"Ж ~{round(profile.get('target_fat', 0))} г, "
        f"У ~{round(profile.get('target_carbs', 0))} г)",
        f"Особенности здоровья: {profile.get('health_conditions', 'не указаны')}",
        f"Аллергии/непереносимости: {profile.get('allergies', 'не указаны')}",
        f"Сон и стресс (по анкете): {profile.get('lifestyle', 'не указаны')}",
        f"Съедено за сегодня ({day}): {round(total_cal)} ккал "
        f"(Б {round(total_p)} г, Ж {round(total_f)} г, У {round(total_c)} г), приёмов пищи: {len(entries)}, вода: {water} мл",
    ]
    if checkin:
        lines.append(
            f"Утренний чек-ин сегодня: сон='{checkin.get('sleep', '')}', "
            f"настрой='{checkin.get('mood', '')}', "
            f"готовность следовать привычке='{checkin.get('readiness', '')}'"
        )
    return "\n".join(lines)


# =============== GEMINI API: СХЕМЫ И ВЫЗОВЫ ===============

FOOD_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "grams": {"type": "number"},
                    "calories": {"type": "number"},
                    "protein": {"type": "number"},
                    "fat": {"type": "number"},
                    "carbs": {"type": "number"},
                },
                "required": ["name", "grams", "calories", "protein", "fat", "carbs"],
            },
        },
        "total_calories": {"type": "number"},
        "total_protein": {"type": "number"},
        "total_fat": {"type": "number"},
        "total_carbs": {"type": "number"},
        "need_clarification": {"type": "boolean"},
        "clarification_question": {"type": "string"},
        "nutritionist_tip": {"type": "string"},
    },
    "required": [
        "description",
        "items",
        "total_calories",
        "total_protein",
        "total_fat",
        "total_carbs",
        "need_clarification",
        "clarification_question",
    ],
}

CHECKIN_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "needs_trainer_attention": {"type": "boolean"},
        "readiness_score": {"type": "integer"},
    },
    "required": ["reply", "needs_trainer_attention", "readiness_score"],
}


def _extract_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Не удалось найти JSON в ответе модели: {text!r}")
        return json.loads(text[start : end + 1])


async def analyze_food_image(image_bytes: bytes, user_id: Optional[int] = None):
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    profile_ctx = format_profile_context(user_id) if user_id else ""

    system_prompt = f"""Ты — профессиональный и бережный ассистент-нутрициолог.
Пользователь прислал фото своей еды или фото этикетки с таблицей КБЖУ.

Контекст пользователя:
{profile_ctx}

Твоя задача:
1. Определи, что на картинке:
   - тарелка с блюдом/блюдами и напитками
   - или этикетка/упаковка с таблицей КБЖУ.
2. Если это ТАРЕЛКА / ЕДА:
   - Определи основные компоненты (белковая часть, гарнир, овощи/салат, соусы/масло, десерты, напитки).
   - Для каждого оцени примерный вес порции в граммах и КБЖУ.
   - Ориентируйся на реалистичные справочные значения на 100 г готового продукта:
     • куриная грудка/индейка готовая ~165 ккал, Б 31 г, Ж 3.5 г, У 0 г
     • говядина тушёная/запечённая ~200–220 ккал, Б 26 г, Ж 12 г, У 0 г
     • рыба белая ~100–130 ккал, красная (лосось/форель) ~200–220 ккал
     • яйца куриные (1 шт ~55 г) ~85 ккал, Б 7 г, Ж 6 г, У 0.5 г
     • крупы отварные (рис, гречка, овсянка на воде) ~110–135 ккал, Б 3–4 г, Ж 1–2 г, У 22–28 г
     • макароны отварные ~150 ккал, Б 5 г, Ж 1.5 г, У 30 г
     • картофель отварной/запечённый ~85–100 ккал, жареный/фри ~220–280 ккал
     • салат из свежих овощей без заправки ~20–25 ккал, с маслом/заправкой +70–120 ккал на порцию
     • растительное масло ~900 ккал/100 г (1 ст. л. 10 г = ~90 ккал, 10 г жира)
3. Если это ЭТИКЕТКА:
   - Прочитай таблицу КБЖУ.
   - Если понятен вес порции (батончик, стаканчик, упаковка) — посчитай на порцию; иначе оцени стандартную порцию и поставь need_clarification=true.
4. Посчитай суммарные total_calories, total_protein, total_fat, total_carbs.
5. В поле nutritionist_tip дай короткий (1–2 предложения) доброжелательный комментарий по тарелке с учётом цели и здоровья/аллергий пользователя (например, похвали за белок/клетчатку или мягко подскажи, чем дополнить).
6. Если состав или размер порции неочевиден, поставь need_clarification=true и напиши конкретный вопрос в clarification_question.
7. Если на фото вообще нет еды и нет этикетки с КБЖУ, верни пустой список items=[] и нулевые значения.
"""

    raw_text = await create_interaction(
        model=GEMINI_MODEL,
        system_instruction=system_prompt,
        input_data=[
            {
                "type": "text",
                "text": "Проанализируй это фото и верни строго JSON по схеме.",
            },
            {
                "type": "image",
                "data": img_b64,
                "mime_type": "image/jpeg",
            },
        ],
        response_schema=FOOD_ANALYSIS_SCHEMA,
    )

    data = _extract_json(raw_text or "{}")

    description = str(data.get("description", "")).strip()
    items_raw = data.get("items", [])
    total_calories = float(data.get("total_calories", 0) or 0)
    total_protein = float(data.get("total_protein", 0) or 0)
    total_fat = float(data.get("total_fat", 0) or 0)
    total_carbs = float(data.get("total_carbs", 0) or 0)
    need_clarification = bool(data.get("need_clarification", False))
    clarification_question = str(data.get("clarification_question", "") or "").strip()
    nutritionist_tip = str(data.get("nutritionist_tip", "") or "").strip()

    items = []
    for item in items_raw:
        items.append(
            {
                "name": str(item.get("name", "Блюдо")),
                "grams": float(item.get("grams", 0) or 0),
                "calories": float(item.get("calories", 0) or 0),
                "protein": float(item.get("protein", 0) or 0),
                "fat": float(item.get("fat", 0) or 0),
                "carbs": float(item.get("carbs", 0) or 0),
            }
        )

    if items:
        total_calories = sum(i["calories"] for i in items)
        total_protein = sum(i["protein"] for i in items)
        total_fat = sum(i["fat"] for i in items)
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
        nutritionist_tip,
    )


async def analyze_food_text(
    text_description: str,
    base_items: Optional[List[Dict[str, Any]]] = None,
    user_id: Optional[int] = None,
):
    profile_ctx = format_profile_context(user_id) if user_id else ""

    if base_items:
        base_lines = [
            f"- {item.get('name', 'Блюдо')}, ~{round(item.get('grams', 0))} г, "
            f"~{round(item.get('calories', 0))} ккал, "
            f"Б {round(item.get('protein', 0))} г, Ж {round(item.get('fat', 0))} г, У {round(item.get('carbs', 0))} г"
            for item in base_items
        ]
        base_text = "\n".join(base_lines)
        user_prompt = f"""Вот что было посчитано в приёме пищи изначально:
{base_text}

Пользователь прислал уточнение или исправление:
\"\"\"{text_description}\"\"\"

Пойми, какие позиции нужно добавить, изменить по массе/составу или удалить, и верни полный итоговый состав этого приёма пищи."""
    else:
        user_prompt = f"""Пользователь описывает свой приём пищи текстом или голосом:
\"\"\"{text_description}\"\"\"

Разбей приём пищи на отдельные продукты/блюда, оцени реалистичную массу в граммах (если она не указана явно) и посчитай КБЖУ."""

    system_prompt = f"""Ты — профессиональный и бережный ассистент-нутрициолог.
Контекст пользователя:
{profile_ctx}

Считай калории и КБЖУ реалистично, без завышения и занижения.
В поле nutritionist_tip добавь короткий тёплый комментарий (1–2 предложения) по этому приёму пищи с учётом целей и особенностей пользователя."""

    raw_text = await create_interaction(
        model=GEMINI_MODEL,
        system_instruction=system_prompt,
        input_data=user_prompt,
        response_schema=FOOD_ANALYSIS_SCHEMA,
    )

    data = _extract_json(raw_text or "{}")

    description = str(data.get("description", "")).strip()
    items_raw = data.get("items", [])
    nutritionist_tip = str(data.get("nutritionist_tip", "") or "").strip()

    items = []
    for item in items_raw:
        items.append(
            {
                "name": str(item.get("name", "Блюдо")),
                "grams": float(item.get("grams", 0) or 0),
                "calories": float(item.get("calories", 0) or 0),
                "protein": float(item.get("protein", 0) or 0),
                "fat": float(item.get("fat", 0) or 0),
                "carbs": float(item.get("carbs", 0) or 0),
            }
        )

    total_calories = sum(i["calories"] for i in items)
    total_protein = sum(i["protein"] for i in items)
    total_fat = sum(i["fat"] for i in items)
    total_carbs = sum(i["carbs"] for i in items)

    return (
        description,
        items,
        total_calories,
        total_protein,
        total_fat,
        total_carbs,
        nutritionist_tip,
    )


async def transcribe_voice(voice_bytes: bytes) -> str:
    """Распознаёт голосовое сообщение через Gemini 3.5 Transcribe (с фоллбэком на Gemini 3.8 Flash)."""
    audio_b64 = base64.b64encode(voice_bytes).decode("utf-8")
    try:
        text = await create_interaction(
            model=GEMINI_TRANSCRIBE_MODEL,
            input_data=[
                {
                    "type": "audio",
                    "data": audio_b64,
                    "mime_type": "audio/ogg",
                }
            ],
        )
        if text:
            return text
    except Exception as e:
        logger.warning("Фоллбэк распознавания голоса через %s: %s", GEMINI_MODEL, e)

    try:
        return await create_interaction(
            model=GEMINI_MODEL,
            input_data=[
                {
                    "type": "text",
                    "text": "Распознай речь из этого аудио дословно на русском языке. Верни только сам распознанный текст без кавычек и пояснений.",
                },
                {
                    "type": "audio",
                    "data": audio_b64,
                    "mime_type": "audio/ogg",
                },
            ],
        )
    except Exception as e:
        logger.exception("Ошибка при распознавании голоса: %s", e)
        return ""


# =============== ЧЕК-ИН НАСТРОЯ, СНА И ГОТОВНОСТИ ===============

async def evaluate_daily_checkin(
    user_id: int,
    sleep_text: str,
    mood_text: str,
    readiness_text: str,
) -> Dict[str, Any]:
    """
    Анализирует утренний/дневной чек-ин пользователя (сон, настрой, готовность следовать привычке).
    Если есть трудности — поддерживает и мягко направляет к тренеру за беседой.
    """
    profile_ctx = format_profile_context(user_id)

    system_prompt = f"""Ты — заботливый бот-нутрициолог и коуч по пищевым привычкам.
Твоя задача — проанализировать ежедневный чек-ин клиента:
1) Как человек спал;
2) Какой у него сегодня настрой и эмоциональное/физическое состояние;
3) Насколько он сегодня готов снова следовать выбранной привычке питаться.

Данные клиента:
{profile_ctx}

Правила ответа (в поле reply):
- Обращайся по имени, пиши тепло, живо и бережно (на русском языке, без сухого канцелярита).
- Обязательно отреагируй и на сон, и на настрой, и на готовность следовать привычке питания.
- Если человек спал хорошо, в хорошем настроении и готов следовать привычке (оценка 7–10 или уверенный настрой):
  • искренне поддержи его настрой;
  • дай 1 конкретный маленький фокус на сегодня (например: начать с белкового завтрака, пить воду, присылать фото тарелок);
  • поставь needs_trainer_attention = false.
- Если есть ЛЮБЫЕ сложности или тревожные сигналы (недосып, усталость, стресс, тревога, плохое настроение, сомнения в своих силах, готовность ниже 7 из 10, срыв, вина, физический дискомфорт):
  • нормализуй состояние: объясни, что на фоне недосыпа или стресса организму физиологически сложнее держать контроль и тянет на быстрые углеводы — в этом нет вины человека;
  • предложи снизить планку на сегодня до «программы-минимум» (просто регулярно есть тёплую сытную еду и не голодать);
  • ОБЯЗАТЕЛЬНО мягко и ненавязчиво предложи написать {TRAINER_CONTACT} за поддерживающей беседой (например: «И пожалуйста, не оставайся с этим один на один — напиши сегодня {TRAINER_CONTACT}, чтобы просто пообщаться и вместе мягко подстроить план под твоё сегодняшнее состояние 💛»);
  • поставь needs_trainer_attention = true.
"""

    user_input = (
        f"Ответы на сегодняшний чек-ин:\n"
        f"1. Сон: {sleep_text}\n"
        f"2. Настрой и состояние: {mood_text}\n"
        f"3. Готовность следовать привычке питаться сегодня: {readiness_text}"
    )

    raw_text = await create_interaction(
        model=GEMINI_MODEL,
        system_instruction=system_prompt,
        input_data=user_input,
        response_schema=CHECKIN_ANALYSIS_SCHEMA,
    )

    data = _extract_json(raw_text or "{}")
    return {
        "reply": str(data.get("reply", "")).strip(),
        "needs_trainer_attention": bool(data.get("needs_trainer_attention", False)),
        "readiness_score": int(data.get("readiness_score", 7) or 7),
    }


# =============== ЧАТ С НУТРИЦИОЛОГОМ-КОУЧЕМ ===============

async def chat_with_coach(user_id: int, user_text: str) -> str:
    history = user_chat_history[user_id]
    profile_ctx = format_profile_context(user_id)
    day = get_active_day(user_id)
    has_checkin_today = day in user_checkins.get(user_id, {})

    checkin_hint = (
        "Сегодня пользователь уже проходил чек-ин (данные есть в контексте выше)."
        if has_checkin_today
        else "Сегодня пользователь ещё НЕ проходил чек-ин: при случае мягко поинтересуйся, как он спал, какой настрой на день и насколько он готов сегодня следовать своей привычке питаться."
    )

    system_instruction = f"""Ты — доброжелательный, чуткий бот-нутрициолог и психолог пищевого поведения.

Контекст и профиль клиента:
{profile_ctx}

Статус чек-ина:
{checkin_hint}

Твоя зона и задачи:
1. Помогать формировать устойчивую привычку питаться без жёстких диет, запретов и чувства вины.
2. Учитывать особенности здоровья, аллергии и образ жизни клиента из анкеты.
3. Интересоваться состоянием человека в комплексе:
   — как он сегодня спал и выспался ли;
   — какой у него настрой на день и уровень стресса/энергии;
   — насколько он сегодня готов снова следовать выбранной привычке питаться.
4. Как отвечать:
   — простым, тёплым, поддерживающим русским языком (без звёздочек Markdown **);
   — нормализуй трудности: «сорвалась», «переела вечером», «нет сил готовить» — это сигнал усталости, недосыпа или голода днём, а не слабости воли;
   — предлагай один маленький реалистичный следующий шаг на сегодня;
   — не ставь медицинских диагнозов и не назначай лекарства.
5. ВАЖНОЕ ПРАВИЛО ПЕРЕНАПРАВЛЕНИЯ К ТРЕНЕРУ:
   — Если ты видишь, что у человека есть проблемы: сильный стресс, плохой сон, упадок настроения, срыв, чувство вины, страх не справиться, потеря мотивации или низкая готовность сегодня соблюдать привычку питания — после тёплой поддержки ОБЯЗАТЕЛЬНО мягко предложи ему написать {TRAINER_CONTACT} за личной беседой (например: «Если чувствуешь, что сегодня тяжело или нужна живая опора — напиши {TRAINER_CONTACT}, обсудите это в беседе и вместе подстроите комфортный темп на ближайшие дни»).
   — Если звучат мысли о самоповреждении или тяжёлом кризисе — бережно предложи обратиться к врачу/психотерапевту и к своему тренеру.
6. Если разговор уходит совсем далеко от питания, сна, состояния, привычек и заботы о себе — тепло возвращай к самочувствию и питанию.
"""

    history_lines = []
    for msg in history[-10:]:
        role_label = "Клиент" if msg.get("role") == "user" else "Нутрициолог"
        history_lines.append(f"{role_label}: {msg.get('content', '')}")

    if history_lines:
        prompt_input = (
            "Предыдущий контекст беседы:\n"
            + "\n".join(history_lines)
            + f"\n\nНовое сообщение клиента:\n{user_text}"
        )
    else:
        prompt_input = user_text

    reply = await create_interaction(
        model=GEMINI_MODEL,
        system_instruction=system_instruction,
        input_data=prompt_input,
    )
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    user_chat_history[user_id] = history[-20:]
    save_storage()

    return reply


# =============== ИСТОРИЯ И САММАРИ ===============

def aggregate_day(user_id: int, day: str):
    entries = user_food_log[user_id].get(day, [])
    total_cal = sum(float(e.get("calories", 0) or 0) for e in entries)
    total_p = sum(float(e.get("protein", 0) or 0) for e in entries)
    total_f = sum(float(e.get("fat", 0) or 0) for e in entries)
    total_c = sum(float(e.get("carbs", 0) or 0) for e in entries)
    return total_cal, total_p, total_f, total_c, entries


async def send_day_history(update: Update, user_id: int, day: str):
    total_cal, total_p, total_f, total_c, entries = aggregate_day(user_id, day)
    water = user_water_log[user_id].get(day, 0)
    checkin = user_checkins.get(user_id, {}).get(day)

    if not entries and water == 0 and not checkin:
        await safe_reply(
            update,
            f"За {day} пока нет записей 🍽\n"
            "Пришли фото еды, напиши текстом/голосом что съел(а) или нажми «☀️ Настрой на день».",
            reply_markup=get_main_keyboard(),
        )
        return

    lines = [f"📜 *История за {day}:*\n"]

    if checkin:
        lines.append(
            "☀️ *Чек-ин дня:*\n"
            f"• Сон: {checkin.get('sleep', '—')}\n"
            f"• Настрой: {checkin.get('mood', '—')}\n"
            f"• Готовность к привычке: {checkin.get('readiness', '—')}\n"
        )

    if entries:
        lines.append("🍽 *Приёмы пищи:*")
        for i, e in enumerate(entries, start=1):
            grams = e.get("grams", 0)
            grams_str = f" (~{round(grams)} г)" if grams else ""
            lines.append(
                f"{i}. {e['description']}{grams_str} — ~{round(e['calories'])} ккал, "
                f"Б {round(e['protein'])} г, Ж {round(e['fat'])} г, У {round(e['carbs'])} г"
            )
        lines.append(
            f"\n📊 Итого за день: *{round(total_cal)} ккал*, "
            f"Б *{round(total_p)} г*, Ж *{round(total_f)} г*, У *{round(total_c)} г*"
        )
    else:
        lines.append("🍽 Записей о еде за этот день пока нет.")

    if water > 0:
        lines.append(f"💧 Вода: *{water} мл*")

    profile = user_profiles.get(user_id)
    if profile and profile.get("target_calories"):
        lines.append(
            f"\n🎯 Твоя цель: ~{round(profile['target_calories'])} ккал "
            f"(Б ~{round(profile.get('target_protein', 0))} г, "
            f"Ж ~{round(profile.get('target_fat', 0))} г, "
            f"У ~{round(profile.get('target_carbs', 0))} г)"
        )

    await safe_reply(
        update,
        "\n".join(lines),
        reply_markup=get_main_keyboard(),
    )


async def week_history(update: Update, user_id: int):
    today = date.today()
    lines = ["📆 *Как ты ел(а) за последние 7 дней:*\n"]
    profile = user_profiles.get(user_id)
    target = profile.get("target_calories") if profile else None

    days_with_food = 0
    week_cal = week_p = week_f = week_c = 0.0

    for i in range(6, -1, -1):
        d = (today - timedelta(days=i)).isoformat()
        total_cal, total_p, total_f, total_c, _ = aggregate_day(user_id, d)
        checkin = user_checkins.get(user_id, {}).get(d)
        checkin_badge = " ☀️" if checkin else ""

        if total_cal == 0:
            lines.append(f"• {d}{checkin_badge}: нет записей еды")
        else:
            days_with_food += 1
            week_cal += total_cal
            week_p += total_p
            week_f += total_f
            week_c += total_c
            if target:
                diff = total_cal - target
                if abs(diff) <= target * 0.12:
                    mark = "✅ в диапазоне"
                elif diff < 0:
                    mark = "⚠️ ниже цели"
                else:
                    mark = "⚠️ выше цели"
                lines.append(
                    f"• {d}{checkin_badge}: ~{round(total_cal)} ккал "
                    f"(Б {round(total_p)} / Ж {round(total_f)} / У {round(total_c)}) — {mark}"
                )
            else:
                lines.append(
                    f"• {d}{checkin_badge}: ~{round(total_cal)} ккал "
                    f"(Б {round(total_p)} / Ж {round(total_f)} / У {round(total_c)})"
                )

    if days_with_food > 0:
        avg_cal = week_cal / days_with_food
        avg_p = week_p / days_with_food
        avg_f = week_f / days_with_food
        avg_c = week_c / days_with_food
        lines.append(
            f"\n📈 *Среднее за день (по {days_with_food} дн. с записями):*\n"
            f"~{round(avg_cal)} ккал | Б {round(avg_p)} г | Ж {round(avg_f)} г | У {round(avg_c)} г"
        )

    lines.append(
        "\nЕсли хочешь разобрать, как прошла неделя, или что-то даётся непросто — "
        "просто напиши мне в чат или обсуди это со своим тренером 💛"
    )

    await safe_reply(update, "\n".join(lines), reply_markup=get_main_keyboard())


# =============== ЗАПИСЬ И ИСПРАВЛЕНИЕ ЕДЫ ===============

def _append_items_to_day(user_id: int, day: str, items: List[Dict[str, Any]]) -> int:
    entries = user_food_log[user_id][day]
    before_len = len(entries)
    for item in items:
        entries.append(
            {
                "description": item.get("name", "Блюдо"),
                "grams": float(item.get("grams", 0) or 0),
                "calories": float(item.get("calories", 0) or 0),
                "protein": float(item.get("protein", 0) or 0),
                "fat": float(item.get("fat", 0) or 0),
                "carbs": float(item.get("carbs", 0) or 0),
            }
        )
    return len(entries) - before_len


def _format_meal_items_block(
    header: str,
    desc: str,
    items: List[Dict[str, Any]],
    total_cal: float,
    total_p: float,
    total_f: float,
    total_c: float,
    nutritionist_tip: str = "",
) -> List[str]:
    lines = [f"<b>{html.escape(header.strip())}</b>\n"]
    if desc:
        lines.append(f"📋 <i>{html.escape(desc)}</i>\n")

    for item in items:
        name = item.get("name", "Блюдо")
        grams = item.get("grams", 0)
        cal = item.get("calories", 0)
        p = item.get("protein", 0)
        f = item.get("fat", 0)
        c = item.get("carbs", 0)
        if grams:
            lines.append(
                f"• <b>{html.escape(name)}</b> (~{round(grams)} г) — <b>{round(cal)} ккал</b>\n"
                f"  <code>Б: {round(p)}г | Ж: {round(f)}г | У: {round(c)}г</code>"
            )
        else:
            lines.append(
                f"• <b>{html.escape(name)}</b> — <b>{round(cal)} ккал</b>\n"
                f"  <code>Б: {round(p)}г | Ж: {round(f)}г | У: {round(c)}г</code>"
            )

    lines.append(
        f"\n🔢 <b>Итого за приём:</b> <b>~{round(total_cal)} ккал</b>\n"
        f"<code>Б: {round(total_p)}г  Ж: {round(total_f)}г  У: {round(total_c)}г</code>"
    )
    if nutritionist_tip:
        lines.append(f"\n<blockquote>💡 <b>Совет нутрициолога:</b>\n{html.escape(nutritionist_tip)}</blockquote>")
    return lines


async def log_food_from_text(update: Update, user_id: int, text: str):
    """Записывает новый приём пищи по текстовому или голосовому описанию без фото."""
    state = user_states.setdefault(user_id, {})
    day = get_active_day(user_id)

    try:
        desc, items, total_cal, total_p, total_f, total_c, tip = await analyze_food_text(
            text, base_items=None, user_id=user_id
        )
    except Exception as e:
        logger.exception("Ошибка при расчёте еды по тексту: %s", e)
        await safe_reply(
            update,
            "Сервис анализа сейчас немного перегружен или обрабатывает запрос дольше обычного 😔\n"
            "Пожалуйста, попробуй отправить сообщение ещё раз через пару секунд 🙏",
        )
        return

    if not items:
        await safe_reply(
            update,
            "Не удалось выделить блюда из описания 🤔\n"
            "Напиши, пожалуйста, чуть конкретнее — например: «овсянка на молоке 200 г и банан».",
            reply_markup=get_main_keyboard(),
        )
        return

    added = _append_items_to_day(user_id, day, items)
    state["pending_meal"] = {"day": day, "count": added, "items": items}
    state["awaiting_correction"] = False
    state["awaiting_food_text"] = False
    save_storage()

    lines = _format_meal_items_block(
        "🍽 Записала приём пищи в дневник:\n",
        desc,
        items,
        total_cal,
        total_p,
        total_f,
        total_c,
        tip,
    )
    meal_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✏️ Уточнить порцию", callback_data="meal_correct"),
                InlineKeyboardButton("📊 Итог дня", callback_data="show_summary"),
            ]
        ]
    )
    await safe_reply(update, "\n".join(lines), reply_markup=meal_kb, parse_mode=ParseMode.HTML)


async def handle_correction(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    pending = state.get("pending_meal")

    clean_text = strip_correction_prefix(text)
    if not clean_text:
        await safe_reply(
            update,
            "Опиши, пожалуйста, словами, что именно было в тарелке или порции — "
            "например: «гречка 150 г, говядина 120 г, салат без масла».",
            reply_markup=get_main_keyboard(),
        )
        return

    # Если предыдущего приёма пищи нет, просто запишем это как новый приём пищи
    if not pending:
        await log_food_from_text(update, user_id, clean_text)
        return

    base_items = pending.get("items")

    try:
        desc, items, total_cal, total_p, total_f, total_c, tip = await analyze_food_text(
            clean_text, base_items=base_items, user_id=user_id
        )
    except Exception as e:
        logger.exception("Ошибка при пересчёте по тексту: %s", e)
        await safe_reply(
            update,
            "Сервис анализа сейчас немного перегружен или обрабатывает запрос дольше обычного 😔\n"
            "Пожалуйста, попробуй отправить уточнение ещё раз через пару секунд 🙏",
        )
        return

    day = pending["day"]
    old_count = pending.get("count", 0)
    entries = user_food_log[user_id][day]

    if old_count > 0 and len(entries) >= old_count:
        del entries[-old_count:]

    added = _append_items_to_day(user_id, day, items)
    state["pending_meal"] = {"day": day, "count": added, "items": items}
    state["awaiting_correction"] = False
    save_storage()

    lines = _format_meal_items_block(
        "✅ Пересчитала приём пищи с учётом твоих правок:\n",
        desc,
        items,
        total_cal,
        total_p,
        total_f,
        total_c,
        tip,
    )
    lines.append("\n✅ <b>Записи за этот день обновлены!</b>")
    meal_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✏️ Ещё уточнить", callback_data="meal_correct"),
                InlineKeyboardButton("📊 Итог дня", callback_data="show_summary"),
            ]
        ]
    )
    await safe_reply(update, "\n".join(lines), reply_markup=meal_kb, parse_mode=ParseMode.HTML)


# =============== КОМАНДЫ БОТА ===============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    state = user_states.setdefault(user_id, {})
    day = get_active_day(user_id)

    profile = user_profiles.get(user_id)
    if not profile or "target_calories" not in profile:
        state["step"] = "ask_name"
        save_storage()
        await safe_reply(
            update,
            "Привет! Я твой бот-нутрициолог и бережный помощник по привычкам питания 🥦✨\n\n"
            "Я умею:\n"
            "• считать КБЖУ по фото тарелки, этикеткам, голосу и тексту;\n"
            "• каждый день отслеживать твой сон, настрой и готовность следовать привычке;\n"
            "• поддерживать без жёстких диет и вовремя подсказывать, когда стоит пообщаться со своим тренером.\n\n"
            "Давай познакомимся и настроим твой профиль. Как тебя зовут?",
            reply_markup=get_main_keyboard(),
        )
        return

    ensure_profile_macros(profile)
    target_cal = profile.get("target_calories")
    tp = profile.get("target_protein")
    tf = profile.get("target_fat")
    tc = profile.get("target_carbs")
    activity_desc = profile.get("activity_description")
    has_checkin = day in user_checkins.get(user_id, {})

    text_lines = [
        f"Снова привет, {profile.get('name', 'друг')}! 👋💚",
        "Твои ориентиры на день:",
    ]
    if activity_desc:
        text_lines.append(f"• Активность: {activity_desc}")
    if target_cal:
        text_lines.append(f"• Целевой калораж: *~{round(target_cal)} ккал/день*")
    if tp and tf and tc:
        text_lines.append(
            f"• БЖУ: Белки *~{round(tp)} г*, Жиры *~{round(tf)} г*, Углеводы *~{round(tc)} г*"
        )
    text_lines.append(
        "\n📸 Присылай фото еды/этикетки или нажми «🍽 Записать еду», чтобы записать приём пищи текстом или голосом."
    )
    if not has_checkin:
        text_lines.append(
            "\n☀️ И давай начнём день с короткого чек-ина! Нажми кнопку *«☀️ Настрой на день»* (или /checkin), "
            "чтобы отметить, как ты спал(а) и какой сегодня настрой."
        )

    await safe_reply(
        update,
        "\n".join(text_lines),
        reply_markup=get_main_keyboard(),
    )


async def checkin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запускает ежедневный опрос о сне,к настрое на день и готовности следовать привычке."""
    user_id = update.effective_user.id
    if user_id not in user_profiles:
        await safe_reply(
            update,
            "Давай сначала познакомимся 🙂 Нажми /start и заполни короткую анкету.",
            reply_markup=get_main_keyboard(),
        )
        return

    state = user_states.setdefault(user_id, {})
    state["step"] = "checkin_sleep"
    state["awaiting_correction"] = False
    state["awaiting_food_text"] = False
    save_storage()

    name = user_profiles[user_id].get("name", "друг")
    await safe_reply(
        update,
        f"☀️ {name}, давай сделаем короткий чек-ин на сегодня (3 быстрых вопроса)!\n\n"
        "1️⃣ *Как ты сегодня спал(а)?*\n"
        "Сколько примерно часов получилось поспать и удалось ли выспаться?",
        reply_markup=get_main_keyboard(),
    )


async def addfood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_profiles:
        await safe_reply(
            update,
            "Давай сначала познакомимся 🙂 Нажми /start, чтобы заполнить анкету.",
            reply_markup=get_main_keyboard(),
        )
        return

    state = user_states.setdefault(user_id, {})
    state["awaiting_food_text"] = True
    state["awaiting_correction"] = False
    save_storage()

    await safe_reply(
        update,
        "🍽 Напиши или наговори голосом, что ты съел(а) или выпил(а) — "
        "например: _«На завтрак 2 яйца, тост из цельнозернового хлеба и капучино 250 мл»_.\n"
        "А если удобнее — просто пришли фото тарелки или этикетки!",
        reply_markup=get_main_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update,
        "🤖 *Что я умею (на базе Gemini):*\n\n"
        "• *☀️ Настрой на день (/checkin)* — спрашиваю про твой сон, самочувствие и насколько ты готов(а) сегодня следовать своей привычке питания. Если день непростой — помогу снизить нагрузку и подскажу, когда лучше написать своему тренеру.\n"
        "• *📷 Фото еды и этикеток* — распознаю блюда на тарелке или таблицу КБЖУ и считаю калории, белки, жиры и углеводы.\n"
        "• *🍽 Записать еду (/addfood)* — считаю КБЖУ по обычному текстовому или голосовому описанию.\n"
        "• *✏️ Исправления* — если нужно уточнить состав после фото, просто напиши «исправь: там было 150 г курицы без соуса».\n"
        "• *💧 Вода (/water 250)* — веду учёт выпитой воды за день.\n"
        "• *📊 Саммари (/summary), История (/dayhistory) и Неделя (/week)* — показываю наглядную статистику и помогаю разобрать недоедание или переедание.\n"
        "• *💬 Поддержка и психология питания* — пиши или записывай голосовые про эмоции, срывы и усталость.\n\n"
        "Полезные команды:\n"
        "/start — профиль и цели\n"
        "/checkin — чек-ин сна, настроя и готовности\n"
        "/addfood — записать еду текстом или голосом\n"
        "/water 250 — добавить воду (в мл)\n"
        "/summary — итог за сегодня\n"
        "/dayhistory — история за выбранный день\n"
        "/week — сводка за 7 дней\n"
        "/trainer — написать своему тренеру\n"
        "/reset — сбросить анкету и начать заново",
        reply_markup=get_main_keyboard(),
    )


async def trainer_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет карточку для быстрого перехода к тренеру с предзаполненным сообщением."""
    await safe_reply(
        update,
        "Если сегодня непростой день, тяжело соблюдать привычки или нужна живая поддержка и разговор — нажми кнопку ниже.\n\n"
        "В диалоге с тренером сразу откроется готовое сообщение, останется только нажать «Отправить» 💛",
        reply_markup=get_trainer_inline_keyboard(),
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_states.pop(user_id, None)
    user_food_log.pop(user_id, None)
    user_water_log.pop(user_id, None)
    user_checkins.pop(user_id, None)
    user_chat_history.pop(user_id, None)
    user_profiles.pop(user_id, None)
    save_profiles()
    save_storage()
    await safe_reply(
        update,
        "Я очистила анкету, цели и историю в этом чате 🔄\n"
        "Нажми /start, чтобы пройти настройку заново.",
        reply_markup=get_main_keyboard(),
    )


async def water_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    day = get_active_day(user_id)

    if context.args:
        try:
            ml = int(context.args[0])
            user_water_log[user_id][day] += ml
            save_storage()
            await safe_reply(
                update,
                f"💧 Добавлено *+{ml} мл* воды!\nИтого за сегодня ({day}): *{user_water_log[user_id][day]} мл*.",
                reply_markup=get_main_keyboard(),
            )
        except ValueError:
            await safe_reply(update, "Пожалуйста, укажи количество воды числом, например: `/water 250`")
        return

    current = user_water_log[user_id].get(day, 0)
    water_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("+250 мл 🥤", callback_data="add_water:250"),
                InlineKeyboardButton("+350 мл 🥛", callback_data="add_water:350"),
                InlineKeyboardButton("+500 мл 🫗", callback_data="add_water:500"),
            ]
        ]
    )
    await safe_reply(
        update,
        f"💧 Сейчас за сегодня записано: *{current} мл* воды.\n"
        "Выбери порцию кнопкой ниже или напиши любое число в чат (например: `300`):",
        reply_markup=water_kb,
    )


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    day = get_active_day(user_id)
    total_cal, total_p, total_f, total_c, entries = aggregate_day(user_id, day)
    water = user_water_log[user_id].get(day, 0)
    checkin = user_checkins.get(user_id, {}).get(day)

    if not entries and water == 0 and not checkin:
        await safe_reply(
            update,
            "За сегодня у тебя пока нет записей 🍽\n\n"
            "• Нажми *«☀️ Настрой»*, чтобы отметить сон и готовность;\n"
            "• Пришли фото еды/этикетки или нажми *«🍽 Добавить еду»*;\n"
            "• Для воды нажми *«💧 +Вода»*.",
            reply_markup=get_main_keyboard(),
        )
        return

    profile = user_profiles.get(user_id)
    if profile:
        ensure_profile_macros(profile)
    target_cal = profile.get("target_calories") if profile else None
    tp = profile.get("target_protein") if profile else None
    tf = profile.get("target_fat") if profile else None
    tc = profile.get("target_carbs") if profile else None

    lines = [f"📊 <b>Сводка за день ({day}):</b>\n"]

    if checkin:
        lines.append(
            f"☀️ <b>Чек-ин:</b> сон — <i>{html.escape(checkin.get('sleep', '—'))}</i>, "
            f"настрой — <i>{html.escape(checkin.get('mood', '—'))}</i>, "
            f"готовность — <i>{html.escape(checkin.get('readiness', '—'))}</i>\n"
        )
    else:
        lines.append(
            "☀️ <i>Чек-ин настроя ещё не пройден — нажми «☀️ Настрой», чтобы коуч знал твоё состояние.</i>\n"
        )

    cal_line = f"🔥 Калории: <b>{round(total_cal)} ккал</b>"
    if target_cal:
        progress = render_progress_bar(total_cal, target_cal)
        cal_line += f" из ~{round(target_cal)} ккал  {progress}"
    lines.append(cal_line)

    if entries:
        lines.append(
            f"➕ С учётом погрешности (+~250 ккал): <b>~{round(total_cal + 250)} ккал</b>"
        )

    if tp and tf and tc:
        lines.append(
            f"🥩 Белки: <b>{round(total_p)} г</b> / ~{round(tp)} г\n"
            f"🧈 Жиры: <b>{round(total_f)} г</b> / ~{round(tf)} г\n"
            f"🥔 Углеводы: <b>{round(total_c)} г</b> / ~{round(tc)} г"
        )
    else:
        lines.append(
            f"Б: <b>{round(total_p)} г</b>, Ж: <b>{round(total_f)} г</b>, У: <b>{round(total_c)} г</b>"
        )

    water_line = f"💧 Вода: <b>{water} мл</b> / ~2000 мл"
    lines.append(water_line)

    if target_cal and entries:
        diff = total_cal - target_cal
        if abs(diff) <= target_cal * 0.12:
            comment = "✅ Отличный баланс! Сегодня ты прямо в своём комфортном коридоре."
        elif diff < 0:
            comment = (
                "💛 Калорий пока меньше целевого ориентира. "
                "Если день продолжается — всё супер, а если вечер — не занижай рацион слишком сильно."
            )
        else:
            comment = (
                "💛 Сегодня вышло чуть выше ориентира. Это не повод для вины! "
                "На аппетит влияют недосып и стресс. Давай разберём это бережно или обсуди с тренером."
            )
        lines.append(f"\n<blockquote>{comment}</blockquote>")

    summary_buttons = []
    if entries:
        summary_buttons.append([
            InlineKeyboardButton("💭 Разобрать рацион с коучем", callback_data="coach_discuss")
        ])
    summary_buttons.append([
        InlineKeyboardButton("+250 мл 🥤", callback_data="add_water:250"),
        InlineKeyboardButton("+350 мл 🥛", callback_data="add_water:350"),
        InlineKeyboardButton("+500 мл 🫗", callback_data="add_water:500"),
    ])
    summary_buttons.append([
        InlineKeyboardButton("💬 Написать тренеру", url=get_trainer_url())
    ])

    markup = InlineKeyboardMarkup(summary_buttons)

    # Попытка сгенерировать графическую карточку в стиле Apple Fitness / Whoop
    try:
        user_name = profile.get("name", "Друг") if profile else "Друг"
        mood_score = checkin.get("readiness_score") if checkin else None
        mood_text = checkin.get("readiness") if checkin else None
        card_buf = render_day_card(
            user_name=user_name,
            day_str=day,
            eaten_cal=total_cal,
            target_cal=target_cal or 2000,
            protein=total_p,
            target_protein=tp or 120,
            fat=total_f,
            target_fat=tf or 65,
            carbs=total_c,
            target_carbs=tc or 200,
            water_ml=water,
            target_water_ml=2000,
            mood_score=mood_score,
            mood_text=mood_text,
            items_count=len(entries),
        )
        await safe_reply_photo(
            update,
            photo=card_buf,
            caption="\n".join(lines),
            reply_markup=markup,
            parse_mode=ParseMode.HTML,
        )
        return
    except Exception as e:
        logger.exception("Ошибка при генерации карточки саммари: %s", e)

    await safe_reply(
        update,
        "\n".join(lines),
        reply_markup=markup,
        parse_mode=ParseMode.HTML,
    )


async def day_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.setdefault(user_id, {})
    state["awaiting_history_day"] = True
    save_storage()

    history_kb = ReplyKeyboardMarkup(
        [
            [KeyboardButton("Сегодня"), KeyboardButton("Вчера")],
            [KeyboardButton("📊 Саммари за день"), KeyboardButton("☀️ Настрой на день")],
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await safe_reply(
        update,
        "За какой день показать подробную историю? 📅\n"
        "Нажми кнопку *«Сегодня»*, *«Вчера»* или напиши дату в формате `ГГГГ-ММ-ДД`.",
        reply_markup=history_kb,
    )


async def week_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await week_history(update, user_id)


# =============== ОБРАБОТКА ФОТО / ГОЛОСА / ТЕКСТА ===============

async def photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_profiles or "target_calories" not in user_profiles[user_id]:
        await safe_reply(
            update,
            "Давай сначала познакомимся, чтобы я знала твои цели по питанию 🙂\n"
            "Нажми /start и заполни короткую анкету.",
            reply_markup=get_main_keyboard(),
        )
        return

    state = user_states.setdefault(user_id, {})
    day = get_active_day(user_id)

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        img_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.exception("Ошибка загрузки фото: %s", e)
        await safe_reply(update, "Не получилось скачать фото 😔 Попробуй ещё раз.")
        return

    caption = (update.message.caption or "").strip()

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
            tip,
        ) = await analyze_food_image(bytes(img_bytes), user_id=user_id)
    except Exception as e:
        logger.exception("Ошибка анализа фото: %s", e)
        await safe_reply(
            update,
            f"Что-то пошло не так при анализе фото через Gemini 😔\nОшибка: {e}",
        )
        return

    if not items:
        await safe_reply(
            update,
            "Я не вижу на этом фото еды или читаемой этикетки с КБЖУ 😔\n"
            "Попробуй сделать снимок ближе или просто опиши приём пищи текстом/голосом!",
            reply_markup=get_main_keyboard(),
        )
        return

    # Если пользователь сразу подписал фото (например: "тут 150 г индейки и гречка без масла"),
    # уточним расчёт с учётом подписи
    if caption:
        try:
            desc, items, total_cal, total_p, total_f, total_c, tip = await analyze_food_text(
                caption, base_items=items, user_id=user_id
            )
            need_clar = False
        except Exception as e:
            logger.warning("Не удалось применить подпись к фото: %s", e)

    added = _append_items_to_day(user_id, day, items)
    state["pending_meal"] = {"day": day, "count": added, "items": items}
    state["awaiting_correction"] = bool(need_clar)
    save_storage()

    lines = _format_meal_items_block(
        "📷 Я распознала на фото:\n",
        desc,
        items,
        total_cal,
        total_p,
        total_f,
        total_c,
        tip,
    )

    if need_clar and clar_q:
        lines.append(
            "\n❓ *Уточни, пожалуйста, чтобы я посчитала точнее:*\n"
            f"{clar_q}\n\n"
            "Просто ответь следующим сообщением (текстом или голосом) — я сразу обновлю этот приём пищи."
        )
    else:
        lines.append(
            "\n✏️ Если хочешь что-то поправить или добавить (например: «исправь: там ещё сосиска» "
            "или «исправь: не было риса») — напиши мне, и я пересчитаю этот приём."
        )

    if day not in user_checkins.get(user_id, {}):
        lines.append(
            "\n<blockquote>☀️ <i>Кстати, ты сегодня ещё не делал(а) чек-ин сна и настроя — нажми «☀️ Настрой», когда будет минутка!</i></blockquote>"
        )

    meal_kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✏️ Уточнить порцию", callback_data="meal_correct"),
                InlineKeyboardButton("📊 Итог дня", callback_data="show_summary"),
            ]
        ]
    )
    await safe_reply(
        update,
        "\n".join(lines),
        reply_markup=meal_kb,
        parse_mode=ParseMode.HTML,
    )


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        voice = update.message.voice or update.message.audio
        file = await voice.get_file()
        voice_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.exception("Ошибка загрузки аудио: %s", e)
        await safe_reply(update, "Не получилось скачать голосовое 😔 Попробуй ещё раз.")
        return

    text = await transcribe_voice(bytes(voice_bytes))
    if not text:
        await safe_reply(
            update,
            "Не смогла распознать голосовое сообщение 😔 Попробуй сказать чуть чётче или напиши текстом.",
        )
        return

    await safe_reply(update, f"🎙 _Распознано:_ «{text}»")
    await text_message(update, context, forced_text=text)


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE, forced_text: str = None):
    user_id = update.effective_user.id
    text = forced_text if forced_text is not None else (update.message.text or "").strip()
    if not text:
        return

    state = user_states.setdefault(user_id, {})
    step = state.get("step")
    lowered = text.strip().lower()

    # ===== КНОПКИ ГЛАВНОГО МЕНЮ (ПОДДЕРЖКА КОРОТКИХ И СТАРЫХ НАЗВАНИЙ) =====
    if text in ("☀️ Настрой на день", "☀️ Настрой") or lowered in ("настрой", "настрой на день", "/checkin"):
        await checkin_command(update, context)
        return
    if text in ("🍽 Записать еду", "🍽 Добавить еду") or lowered in ("записать еду", "добавить еду", "/addfood"):
        await addfood_command(update, context)
        return
    if text in ("💧 +Вода", "💧 Вода") or lowered in ("вода", "+вода", "/water"):
        await water_command(update, context)
        return
    if text in ("📊 Саммари за день", "📊 Итог дня") or lowered in ("итог дня", "саммари", "саммари за день", "/summary"):
        await summary(update, context)
        return
    if text in ("📜 История за день", "📜 История") or lowered in ("история", "история за день", "/dayhistory"):
        await day_history_command(update, context)
        return
    if text in ("📆 Как я ем неделю", "📆 Неделя") or lowered in ("неделя", "как я ем неделю", "/week"):
        await week_history_command(update, context)
        return
    if text in ("💬 Написать тренеру", "💬 Тренер") or lowered in ("написать тренеру", "/trainer", "тренер"):
        await trainer_command(update, context)
        return
    if text in ("❓ Помощь", "❓ Инфо") or lowered in ("помощь", "/help"):
        await help_command(update, context)
        return

    # ===== ВЫБОР ДАТЫ ДЛЯ ИСТОРИИ ЗА ДЕНЬ =====
    if state.get("awaiting_history_day"):
        state["awaiting_history_day"] = False
        save_storage()
        if lowered.startswith("сегодня"):
            day_str = today_str()
        elif lowered.startswith("вчера"):
            day_str = (date.today() - timedelta(days=1)).isoformat()
        else:
            try:
                day_str = date.fromisoformat(text.strip()).isoformat()
            except ValueError:
                await safe_reply(
                    update,
                    "Не поняла дату 🤔 Напиши «сегодня», «вчера» или дату в формате `ГГГГ-ММ-ДД` (например, `2026-10-18`).",
                    reply_markup=get_main_keyboard(),
                )
                return
        await send_day_history(update, user_id, day_str)
        return

    # ===== ШАГИ ЕЖЕДНЕВНОГО ЧЕК-ИНА (СОН -> НАСТРОЙ -> ГОТОВНОСТЬ) =====
    if step == "checkin_sleep":
        state["checkin_draft_sleep"] = text.strip()
        state["step"] = "checkin_mood"
        save_storage()
        await safe_reply(
            update,
            "2️⃣ Принято! А *как твой настрой и самочувствие на сегодня?*\n"
            "Есть ли бодрость или, наоборот, чувствуется усталость, стресс, напряжение?",
            reply_markup=get_main_keyboard(),
        )
        return

    if step == "checkin_mood":
        state["checkin_draft_mood"] = text.strip()
        state["step"] = "checkin_readiness"
        save_storage()
        readiness_kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("💪 9–10: Отлично!", callback_data="checkin_ready:10"),
                    InlineKeyboardButton("👍 7–8: В норме", callback_data="checkin_ready:8"),
                ],
                [
                    InlineKeyboardButton("😔 4–6: Тяжеловато", callback_data="checkin_ready:5"),
                    InlineKeyboardButton("💛 1–3: Нет сил", callback_data="checkin_ready:2"),
                ],
            ]
        )
        await safe_reply(
            update,
            "3️⃣ И последний вопрос чек-ина:\n"
            "*Насколько от 1 до 10 ты сегодня готов(а) снова следовать выбранной привычке питаться?*\n"
            "Нажми кнопку ниже или напиши своими словами, если есть какие-то сложности:",
            reply_markup=readiness_kb,
        )
        return

    if step == "checkin_readiness":
        sleep_ans = state.pop("checkin_draft_sleep", "не указано")
        mood_ans = state.pop("checkin_draft_mood", "не указано")
        readiness_ans = text.strip()
        state["step"] = None
        day = get_active_day(user_id)

        try:
            eval_res = await evaluate_daily_checkin(user_id, sleep_ans, mood_ans, readiness_ans)
            ai_reply = eval_res["reply"]
            needs_trainer = eval_res["needs_trainer_attention"]
            readiness_score = eval_res["readiness_score"]
        except Exception as e:
            logger.exception("Ошибка при анализе чек-ина: %s", e)
            needs_trainer = any(
                w in readiness_ans.lower() or w in mood_ans.lower() or w in sleep_ans.lower()
                for w in ("тяжел", "нет сил", "плохо", "стресс", "не спал", "1", "2", "3", "4", "5")
            )
            readiness_score = 5 if needs_trainer else 8
            ai_reply = (
                "Спасибо, что поделился(ась) своим состоянием на сегодня! ☀️\n"
                + (
                    f"Вижу, что сегодня день непростой. Пожалуйста, будь к себе бережнее, не требуй от себя идеала и обязательно напиши {TRAINER_CONTACT} за беседой и поддержкой 💛"
                    if needs_trainer
                    else "Отличный настрой! Продолжаем спокойно двигаться по плану — жду фото или описание твоих приёмов пищи 🥦"
                )
            )

        user_checkins[user_id][day] = {
            "sleep": sleep_ans,
            "mood": mood_ans,
            "readiness": readiness_ans,
            "readiness_score": readiness_score,
            "needs_trainer": needs_trainer,
            "ai_reply": ai_reply,
        }
        save_storage()

        if needs_trainer:
            await safe_reply(
                update,
                ai_reply,
                reply_markup=get_trainer_inline_keyboard(),
            )
        else:
            await safe_reply(
                update,
                ai_reply,
                reply_markup=get_main_keyboard(),
            )
        return

    # ===== ОЖИДАНИЕ ТЕКСТОВОЙ ЗАПИСИ ЕДЫ ИЛИ ИСПРАВЛЕНИЯ =====
    if state.get("awaiting_food_text"):
        await log_food_from_text(update, user_id, text)
        return

    if state.get("awaiting_correction") and state.get("pending_meal"):
        await handle_correction(update, context, text)
        return

    if is_correction_command(text):
        await handle_correction(update, context, text)
        return

    profile = user_profiles.get(user_id)

    # ===== АНКЕТА НОВОГО ПОЛЬЗОВАТЕЛЯ =====
    if step is None and (profile is None or "target_calories" not in profile):
        state["step"] = "ask_name"
        save_storage()
        await safe_reply(
            update,
            "Давай сначала познакомимся: как тебя зовут?",
            reply_markup=get_main_keyboard(),
        )
        return

    if step == "ask_name":
        name = text.strip()
        user_profiles[user_id] = {"name": name}
        save_profiles()
        state["step"] = "ask_age"
        save_storage()
        await safe_reply(
            update,
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
            await safe_reply(
                update,
                "Возраст должен быть целым числом (например, 28). Попробуй ещё раз 🙂",
            )
            return
        user_profiles[user_id]["age"] = age
        save_profiles()
        state["step"] = "ask_sex"
        save_storage()
        await safe_reply(
            update,
            "Укажи, пожалуйста, пол (`м` или `ж`). Это нужно для формулы метаболизма BMR.",
        )
        return

    if step == "ask_sex":
        sex = text.strip().lower()
        if not (sex.startswith("м") or sex.startswith("ж")):
            await safe_reply(update, "Напиши, пожалуйста, просто `м` или `ж` 🙂")
            return
        user_profiles[user_id]["sex"] = sex
        save_profiles()
        state["step"] = "ask_weight"
        save_storage()
        await safe_reply(
            update,
            "Теперь напиши свой текущий вес в килограммах (например, `70.5`).",
        )
        return

    if step == "ask_weight":
        try:
            weight = parse_float(text)
            if weight <= 0 or weight > 400:
                raise ValueError
        except ValueError:
            await safe_reply(
                update,
                "Вес должен быть числом в килограммах (например, `68` или `68.5`). Попробуй ещё раз.",
            )
            return
        user_profiles[user_id]["weight"] = weight
        save_profiles()
        state["step"] = "ask_height"
        save_storage()
        await safe_reply(
            update,
            "Отлично! Теперь напиши рост в сантиметрах (например, `170`).",
        )
        return

    if step == "ask_height":
        try:
            height = parse_float(text)
            if height <= 0 or height > 250:
                raise ValueError
        except ValueError:
            await safe_reply(
                update,
                "Рост должен быть числом в сантиметрах (например, `165`). Попробуй ещё раз.",
            )
            return
        user_profiles[user_id]["height"] = height
        save_profiles()
        state["step"] = "ask_activity"
        save_storage()
        await safe_reply(
            update,
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
            await safe_reply(
                update,
                "Пожалуйста, напиши число от 1 до 5, которое лучше всего описывает твою активность.",
            )
            return
        activity_desc, activity_factor = parsed
        user_profiles[user_id]["activity_description"] = activity_desc
        user_profiles[user_id]["activity_factor"] = activity_factor
        save_profiles()
        state["step"] = "ask_target_weight"
        save_storage()
        await safe_reply(
            update,
            "К какому весу ты хочешь прийти? Напиши число в килограммах (например, `60`).",
        )
        return

    if step == "ask_target_weight":
        try:
            target_weight = parse_float(text)
            if target_weight <= 0 or target_weight > 400:
                raise ValueError
        except ValueError:
            await safe_reply(
                update,
                "Целевой вес тоже должен быть числом в килограммах 🙂 Попробуй ещё раз.",
            )
            return
        user_profiles[user_id]["target_weight"] = target_weight
        save_profiles()
        state["step"] = "ask_health"
        save_storage()
        await safe_reply(
            update,
            "Есть ли у тебя какие-то особенности здоровья? (например: диабет, гастрит, инсулинорезистентность)\n"
            "Если нет, просто напиши `нет`.",
        )
        return

    if step == "ask_health":
        user_profiles[user_id]["health_conditions"] = text.strip()
        save_profiles()
        state["step"] = "ask_allergies"
        save_storage()
        await safe_reply(
            update,
            "Есть ли у тебя пищевые аллергии или непереносимости? (например: лактоза, глютен, орехи)\n"
            "Если нет, напиши `нет`.",
        )
        return

    if step == "ask_allergies":
        user_profiles[user_id]["allergies"] = text.strip()
        save_profiles()
        state["step"] = "ask_lifestyle"
        save_storage()
        await safe_reply(
            update,
            "Как у тебя обычно со сном и уровнем стресса? (например: «сплю 6 ч, много стресса на работе»)\n"
            "Это поможет мне давать персональные рекомендации.",
        )
        return

    if step == "ask_lifestyle":
        user_profiles[user_id]["lifestyle"] = text.strip()
        save_profiles()
        state["step"] = "ask_months_to_goal"
        save_storage()
        await safe_reply(
            update,
            "За сколько МЕСЯЦЕВ ты бы хотел(а) прийти к своей цели по весу? Например: `4`.",
        )
        return

    if step == "ask_months_to_goal":
        try:
            months = parse_float(text)
            if months <= 0:
                raise ValueError
        except ValueError:
            await safe_reply(
                update,
                "Срок лучше указать числом в месяцах (например, `4` или `6`). Попробуй ещё раз 🙂",
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
        activity_factor = profile.get("activity_factor", 1.375)
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

        goal_text = {
            "loss": "Похудение",
            "gain": "Набор мышечной массы",
            "maintain": "Поддержание веса",
        }[goal_type]

        profile["goal"] = goal_text
        profile["bmr"] = round(bmr, 1)
        profile["goal_type"] = goal_type
        profile["maintenance_calories"] = round(maintenance, 1)
        profile["target_calories"] = round(target_cal, 1)
        profile["daily_deficit"] = round(daily_def, 1)
        profile["realistic_months"] = round(realistic_months, 1)
        profile.update(macros)
        save_profiles()

        state["step"] = None
        get_active_day(user_id)
        save_storage()

        tp = macros["target_protein"]
        tf = macros["target_fat"]
        tc = macros["target_carbs"]
        health = profile.get("health_conditions", "нет")
        allergies = profile.get("allergies", "нет")

        deficit_line = ""
        if goal_type == "loss":
            deficit_line = f"Ориентировочный дефицит: ~{abs(round(daily_def))} ккал/день.\n"
        elif goal_type == "gain":
            deficit_line = f"Ориентировочный профицит: ~{abs(round(daily_def))} ккал/день.\n"

        text_out = (
            "🎯 *Твой персональный план готов!*\n\n"
            "🧮 *Расчёты (Миффлин–Сан Жеор):*\n"
            f"• Базовый метаболизм (BMR): *{round(bmr)} ккал*\n"
            f"• Расход с учётом активности: *{round(maintenance)} ккал*\n\n"
            "📊 *Твои данные и цель:*\n"
            f"• Вес: {profile['weight']} ➔ {profile['target_weight']} кг ({goal_text.lower()})\n"
            f"• Здоровье: {health}\n"
            f"• Аллергии: {allergies}\n\n"
            "🍎 *Целевое КБЖУ на день:*\n"
            f"• Калории: *~{round(target_cal)} ккал*\n"
            f"• Белки: *~{round(tp)} г*\n"
            f"• Жиры: *~{round(tf)} г*\n"
            f"• Углеводы: *~{round(tc)} г*\n\n"
            f"💡 {deficit_line}{plan_comment}\n\n"
            "☀️ *С чего начнём сегодня:*\n"
            "Нажми кнопку *«☀️ Настрой на день»*, чтобы рассказать, как ты сегодня спал(а) "
            "и насколько готов(а) следовать новой привычке питания, или сразу присылай фото/описание еды!"
        )

        await safe_reply(
            update,
            text_out,
            reply_markup=get_main_keyboard(),
        )
        return

    if step is not None:
        await safe_reply(
            update,
            "Давай сначала закончим текущий шаг — ответь, пожалуйста, на предыдущий вопрос 🙂",
        )
        return

    # ===== КНОПКИ БЫСТРОГО РАЗБОРА ПОСЛЕ САММАРИ =====
    if lowered.startswith("обсудить мой прогресс"):
        reply = await chat_with_coach(
            user_id,
            "Помоги мне обсудить мой сегодняшний прогресс по питанию, сну и привычкам.",
        )
        await safe_reply(update, reply, reply_markup=get_main_keyboard())
        return

    if lowered.startswith("обсудить, почему недоел"):
        reply = await chat_with_coach(
            user_id,
            "Я сегодня недоел(а) относительно своей цели по калориям. Помоги мягко разобраться, почему так могло быть и как себе помочь.",
        )
        await safe_reply(update, reply, reply_markup=get_main_keyboard())
        return

    if lowered.startswith("обсудить переедание"):
        reply = await chat_with_coach(
            user_id,
            "Я сегодня вышел(ла) за пределы своего калоража и переел(а). Помоги разобраться без вины, как на это повлияли сон/стресс/настрой и стоит ли обсудить это с тренером.",
        )
        await safe_reply(update, reply, reply_markup=get_main_keyboard())
        return

    # ===== ЗАПИСЬ ЕДЫ ПО ОБЫЧНОМУ ТЕКСТУ/ГОЛОСУ БЕЗ ФОТО =====
    if is_likely_food_entry(text):
        await log_food_from_text(update, user_id, text)
        return

    # ===== ОБЫЧНЫЙ ТЕКСТ: ДИАЛОГ С КОУЧЕМ-НУТРИЦИОЛОГОМ =====
    try:
        reply = await chat_with_coach(user_id, text)
        needs_trainer_prompt = any(
            w in reply.lower()
            for w in ("тренер", "напиши своему тренеру", "бесед", "поддержк")
        ) and any(
            t in text.lower() or t in reply.lower()
            for t in ("тяжел", "устал", "плохо", "не могу", "сорвал", "стресс", "болит", "боле", "вин", "стыд", "тренер")
        )
        markup = get_trainer_inline_keyboard() if needs_trainer_prompt else get_main_keyboard()
        await safe_reply(
            update,
            reply,
            reply_markup=markup,
            parse_mode=None,
        )
    except Exception as e:
        logger.exception("Ошибка в чат-коуче: %s", e)
        await safe_reply(
            update,
            "Я на секунду отвлеклась из-за временной задержки ответа сервиса 😔\n"
            "Напиши мне, пожалуйста, ещё разок — я уже здесь и готова помочь!",
            reply_markup=get_main_keyboard(),
        )


# =============== OБРАБОТКА CALLBACK QUERY (INLINE КНОПКИ) ===============

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data or ""
    state = user_states.setdefault(user_id, {})
    day = get_active_day(user_id)

    # 1. Быстрое добавление воды
    if data.startswith("add_water:"):
        try:
            ml = int(data.split(":", 1)[1])
            user_water_log.setdefault(user_id, {})
            user_water_log[user_id][day] = user_water_log[user_id].get(day, 0) + ml
            save_storage()
            total_water = user_water_log[user_id][day]
            water_kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("+250 мл 🥤", callback_data="add_water:250"),
                        InlineKeyboardButton("+350 мл 🥛", callback_data="add_water:350"),
                        InlineKeyboardButton("+500 мл 🫗", callback_data="add_water:500"),
                    ],
                    [
                        InlineKeyboardButton("📊 Итог дня", callback_data="show_summary"),
                    ],
                ]
            )
            await query.edit_message_text(
                f"💧 Добавлено *+{ml} мл*!\nИтого за сегодня ({day}): *{total_water} мл* 🥤",
                reply_markup=water_kb,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.exception("Ошибка при добавлении воды через inline-кнопку: %s", e)
        return

    # 2. Уточнение порции после записи еды
    if data == "meal_correct":
        state["awaiting_correction"] = True
        save_storage()
        await query.message.reply_text(
            "✏️ Напиши или наговори голосом, что поправить в этой тарелке:\n"
            "_(например: «съела только половину», «вместо говядины была курица», «добавь ложку сметаны»)_",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard(),
        )
        return

    # 3. Показать саммари за день
    if data == "show_summary":
        await summary(update, context)
        return

    # 4. Разобрать рацион с коучем
    if data == "coach_discuss":
        total_cal, total_p, total_f, total_c, entries = aggregate_day(user_id, day)
        water = user_water_log.get(user_id, {}).get(day, 0)
        profile = user_profiles.get(user_id, {})
        target_cal = profile.get("target_calories", 2000)
        prompt = (
            f"Посмотри на мой сегодняшний рацион: съедено {round(total_cal)} ккал из цели {round(target_cal)} ккал, "
            f"воды {water} мл. Дай короткую, бережную и поддерживающую обратную связь: на что обратить внимание завтра?"
        )
        try:
            coach_reply = await chat_with_coach(user_id, prompt)
            await query.message.reply_text(
                f"💭 *Разбор сегодняшнего рациона:*\n\n{coach_reply}",
                reply_markup=get_main_keyboard(),
                parse_mode=None,
            )
        except Exception as e:
            logger.exception("Ошибка coach_discuss: %s", e)
            await query.message.reply_text(
                "Не удалось сформировать разбор прямо сейчас 😔 Попробуй ещё раз чуть позже.",
                reply_markup=get_main_keyboard(),
            )
        return

    # 5. Оценка готовности в чек-ине
    if data.startswith("checkin_ready:"):
        score_val = data.split(":", 1)[1]
        score_labels = {
            "10": "9–10: Полная готовность 💪",
            "8": "7–8: Спокойно продолжаю 👍",
            "5": "4–6: Сегодня тяжеловато 😔",
            "2": "1–3: Нет сил, нужна поддержка 💛",
        }
        readiness_ans = score_labels.get(score_val, f"{score_val} из 10")
        sleep_ans = state.pop("checkin_draft_sleep", "не указано")
        mood_ans = state.pop("checkin_draft_mood", "не указано")
        state["step"] = None

        try:
            eval_res = await evaluate_daily_checkin(user_id, sleep_ans, mood_ans, readiness_ans)
            ai_reply = eval_res["reply"]
            needs_trainer = eval_res["needs_trainer_attention"]
            readiness_score = eval_res["readiness_score"]
        except Exception as e:
            logger.exception("Ошибка при анализе чек-ина через inline-кнопку: %s", e)
            needs_trainer = int(score_val) <= 6
            readiness_score = int(score_val)
            ai_reply = (
                "Спасибо, что поделился(ась) своим состоянием на сегодня! ☀️\n"
                + (
                    f"Вижу, что сегодня день непростой. Пожалуйста, будь к себе бережнее, не требуй от себя идеала и обязательно напиши {TRAINER_CONTACT} за беседой и поддержкой 💛"
                    if needs_trainer
                    else "Отличный настрой! Продолжаем спокойно двигаться по плану — жду фото или описание твоих приёмов пищи 🥦"
                )
            )

        user_checkins.setdefault(user_id, {})[day] = {
            "sleep": sleep_ans,
            "mood": mood_ans,
            "readiness": readiness_ans,
            "readiness_score": readiness_score,
            "needs_trainer": needs_trainer,
            "ai_reply": ai_reply,
        }
        save_storage()

        markup = get_trainer_inline_keyboard() if needs_trainer else get_main_keyboard()
        await query.message.reply_text(ai_reply, reply_markup=markup)
        return


# =============== MAIN ===============

async def post_init(application):
    """Регистрирует нативные команды в Telegram Menu Button."""
    try:
        await application.bot.set_my_commands([
            BotCommand("summary", "📊 Итог дня и баланс"),
            BotCommand("checkin", "☀️ Чек-ин сна и настроя"),
            BotCommand("addfood", "🍽 Добавить еду"),
            BotCommand("water", "💧 Учёт выпитой воды"),
            BotCommand("week", "📆 Сводка за неделю"),
            BotCommand("dayhistory", "📜 История за день"),
            BotCommand("trainer", "💬 Написать тренеру"),
            BotCommand("help", "❓ Справка по боту"),
        ])
        logger.info("Нативные команды меню бота успешно зарегистрированы в Telegram")
    except Exception as e:
        logger.warning("Не удалось зарегистрировать команды меню бота: %s", e)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("checkin", checkin_command))
    app.add_handler(CommandHandler("addfood", addfood_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("week", week_history_command))
    app.add_handler(CommandHandler("dayhistory", day_history_command))
    app.add_handler(CommandHandler("water", water_command))
    app.add_handler(CommandHandler("trainer", trainer_command))

    app.add_handler(CallbackQueryHandler(handle_callback_query))

    app.add_handler(MessageHandler(filters.PHOTO, photo_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

    logger.info("Нутри-бот на базе Gemini запущен (polling). Ждём сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()


