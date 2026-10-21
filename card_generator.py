"""
Генератор стильных инфографик-карточек для нутри-бота «Ам-Ам».
Создаёт карточки в стиле Apple Fitness / Whoop для отчётов за день и неделю.
"""

import io
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def get_font(size: int, bold: bool = False):
    """Возвращает системный шрифт с поддержкой кириллицы."""
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_day_card(
    user_name: str,
    day_str: str,
    eaten_cal: float,
    target_cal: float,
    protein: float,
    target_protein: float,
    fat: float,
    target_fat: float,
    carbs: float,
    target_carbs: float,
    water_ml: int,
    target_water_ml: int = 2000,
    mood_score: int = None,
    mood_text: str = None,
    items_count: int = 0,
) -> io.BytesIO:
    """Генерирует стильную инфографику 1080x1350 за день."""
    w, h = 1080, 1350
    # Премиальный тёмный фон (глубокий графит / тёмный сланец)
    img = Image.new("RGBA", (w, h), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)

    # Шрифты
    f_title = get_font(52, bold=True)
    f_subtitle = get_font(32, bold=False)
    f_card_title = get_font(36, bold=True)
    f_big_num = get_font(84, bold=True)
    f_label = get_font(28, bold=False)
    f_val = get_font(32, bold=True)
    f_badge = get_font(26, bold=True)

    # 1. Фоновые декоративные неоновые круги (мягкое свечение)
    glow_cal = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_cal)
    glow_draw.ellipse([700, -50, 1150, 400], fill=(16, 185, 129, 30))  # изумруд
    glow_draw.ellipse([-100, 600, 400, 1100], fill=(59, 130, 246, 25))  # сапфир
    img = Image.alpha_composite(img, glow_cal)
    draw = ImageDraw.Draw(img)

    # 2. Шапка
    # Форматирование даты
    try:
        dt = datetime.strptime(day_str, "%Y-%m-%d")
        months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        date_formatted = f"{dt.day} {months[dt.month - 1]} {dt.year}"
    except Exception:
        date_formatted = day_str

    draw.text((70, 70), "ДНЕВНИК ПИТАНИЯ", font=f_subtitle, fill=(148, 163, 184))
    draw.text((70, 115), f"Итоги дня: {user_name}", font=f_title, fill=(248, 250, 252))

    # Бейдж даты справа вверху
    badge_w, badge_h = 280, 52
    badge_x = w - 70 - badge_w
    draw.rounded_rectangle([badge_x, 80, badge_x + badge_w, 80 + badge_h], radius=26, fill=(30, 41, 59))
    draw.text((badge_x + 28, 92), date_formatted, font=f_badge, fill=(226, 232, 240))

    # 3. Главная карточка: КАЛОРИИ
    card1_y1, card1_y2 = 210, 540
    draw.rounded_rectangle([70, card1_y1, w - 70, card1_y2], radius=32, fill=(30, 41, 59, 230), outline=(51, 65, 85), width=2)

    draw.text((110, 245), "КАЛОРИИ ЗА ДЕНЬ", font=f_label, fill=(148, 163, 184))
    eaten_str = f"{round(eaten_cal)}"
    draw.text((110, 280), eaten_str, font=f_big_num, fill=(255, 255, 255))
    
    # Текст цели
    target_str = f"из {round(target_cal)} ккал" if target_cal else "ккал"
    draw.text((110 + len(eaten_str) * 52 + 25, 325), target_str, font=f_card_title, fill=(148, 163, 184))

    # Прогресс бар калорий
    pct = min(1.0, eaten_cal / target_cal) if target_cal and target_cal > 0 else 0.0
    pct_text = f"{round(pct * 100)}%"
    
    bar_x1, bar_y1 = 110, 420
    bar_x2, bar_y2 = w - 110, 452
    draw.rounded_rectangle([bar_x1, bar_y1, bar_x2, bar_y2], radius=16, fill=(15, 23, 42))

    fill_w = int((bar_x2 - bar_x1) * pct)
    if fill_w > 16:
        # Градиентный цвет: изумрудный, при переборе - теплый коралл
        bar_color = (16, 185, 129) if pct <= 1.05 else (244, 63, 94)
        draw.rounded_rectangle([bar_x1, bar_y1, bar_x1 + fill_w, bar_y2], radius=16, fill=bar_color)

    # Процент справа над баром
    draw.text((w - 110 - 70, 375), pct_text, font=f_badge, fill=(16, 185, 129) if pct <= 1.05 else (244, 63, 94))

    # Мини-инфо о погрешности
    draw.text((110, 475), f"Зафиксировано приёмов: {items_count}  •  С погрешностью: ~{round(eaten_cal + 250)} ккал", font=f_label, fill=(100, 116, 139))

    # 4. Вторая карточка: МАКРОНУТРИЕНТЫ (БЖУ)
    card2_y1, card2_y2 = 570, 940
    draw.rounded_rectangle([70, card2_y1, w - 70, card2_y2], radius=32, fill=(30, 41, 59, 230), outline=(51, 65, 85), width=2)
    draw.text((110, 605), "БАЛАНС БЖУ (ГРАММЫ)", font=f_label, fill=(148, 163, 184))

    macros = [
        ("Белки", protein, target_protein or 120, (56, 189, 248), 660),     # Cyan
        ("Жиры", fat, target_fat or 65, (251, 191, 36), 750),               # Amber
        ("Углеводы", carbs, target_carbs or 200, (192, 132, 252), 840),     # Purple
    ]

    for name, cur, tgt, color, my in macros:
        draw.text((110, my), name, font=f_val, fill=(241, 245, 249))
        val_str = f"{round(cur)} / {round(tgt)} г"
        draw.text((w - 110 - len(val_str) * 18, my), val_str, font=f_label, fill=(203, 213, 225))

        # Бар
        m_pct = min(1.0, cur / tgt) if tgt and tgt > 0 else 0.0
        m_bx1, m_by1 = 110, my + 45
        m_bx2, m_by2 = w - 110, my + 63
        draw.rounded_rectangle([m_bx1, m_by1, m_bx2, m_by2], radius=9, fill=(15, 23, 42))
        m_fw = int((m_bx2 - m_bx1) * m_pct)
        if m_fw > 10:
            draw.rounded_rectangle([m_bx1, m_by1, m_bx1 + m_fw, m_by2], radius=9, fill=color)

    # 5. Две нижние плашки: ВОДА и НАСТРОЙ (в 2 колонки)
    card3_y1, card3_y2 = 970, 1190
    half_w = (w - 140 - 24) // 2

    # Левая колонка: Вода
    col1_x1 = 70
    col1_x2 = col1_x1 + half_w
    draw.rounded_rectangle([col1_x1, card3_y1, col1_x2, card3_y2], radius=32, fill=(30, 41, 59, 230), outline=(51, 65, 85), width=2)
    draw.text((col1_x1 + 35, card3_y1 + 35), "💧 ВОДА", font=f_label, fill=(148, 163, 184))
    draw.text((col1_x1 + 35, card3_y1 + 75), f"{water_ml}", font=f_big_num, fill=(56, 189, 248))
    draw.text((col1_x1 + 35, card3_y1 + 165), f"из {target_water_ml} мл", font=f_label, fill=(148, 163, 184))

    # Правая колонка: Настрой / Чек-ин
    col2_x1 = col1_x2 + 24
    col2_x2 = w - 70
    draw.rounded_rectangle([col2_x1, card3_y1, col2_x2, card3_y2], radius=32, fill=(30, 41, 59, 230), outline=(51, 65, 85), width=2)
    draw.text((col2_x1 + 35, card3_y1 + 35), "☀️ НАСТРОЙ", font=f_label, fill=(148, 163, 184))

    score_display = f"{mood_score}/10" if mood_score is not None else "—"
    score_color = (16, 185, 129) if (mood_score or 0) >= 7 else (251, 191, 36) if (mood_score or 0) >= 4 else (244, 63, 94)
    draw.text((col2_x1 + 35, card3_y1 + 75), score_display, font=f_big_num, fill=score_color)
    mood_label = mood_text if mood_text else ("Чек-ин пройден" if mood_score else "Чек-ин не пройден")
    if len(mood_label) > 22:
        mood_label = mood_label[:20] + "…"
    draw.text((col2_x1 + 35, card3_y1 + 165), mood_label, font=f_label, fill=(148, 163, 184))

    # 6. Футер
    draw.text((70, 1235), "Ам-Ам • Заботливый ИИ-нутрициолог", font=f_label, fill=(100, 116, 139))
    draw.text((w - 70 - 240, 1235), "Тренер: @VolkovaYasno", font=f_label, fill=(148, 163, 184))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf
