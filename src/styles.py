"""
Theme & Stylesheet Configuration for Sign Language Translator
=============================================================
This file contains the complete visual theme specification.
All colors, card styles, buttons, inputs, sliders, and toggle switches
are defined here and loaded into the main window.
"""

from pathlib import Path

# ==============================================================================
# ACCENT COLOR CONFIGURATION
# Change this single variable to easily update the accent color across the app!
# Examples:
#   Indigo (Default): "#6C63FF"
#   Electric Blue:    "#3B82F6"
#   Cyan:             "#06B6D4"
#   Emerald Green:    "#10B981"
#   Violet:           "#8B5CF6"
#   Rose:             "#F43F5E"
# ==============================================================================
ACCENT_COLOR = "#6C63FF"

# Background & Card Colors
BG_DARK = "#0F1420"
CARD_BG = "#161B2B"
CARD_BORDER = "#232B42"
INPUT_BG = "#0B0E17"
INPUT_BORDER = "#232B42"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"
DANGER_RED = "#EF4444"
DANGER_RED_HOVER = "#DC2626"
DANGER_RED_PRESSED = "#B91C1C"
SUCCESS_GREEN = "#10B981"


def _generate_assets(accent: str, assets_dir: Path):
    """Generates modern SVG assets (toggle switches & chevrons) matching the accent color."""
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Toggle Off
    toggle_off_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="38" height="22" viewBox="0 0 38 22">
  <rect x="1" y="1" width="36" height="20" rx="10" fill="#1C2337" stroke="#2D3748" stroke-width="1.5"/>
  <circle cx="11" cy="11" r="7" fill="#64748B"/>
</svg>"""
    (assets_dir / "toggle_off.svg").write_text(toggle_off_svg, encoding="utf-8")

    # Toggle Off Hover
    toggle_off_hover_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="38" height="22" viewBox="0 0 38 22">
  <rect x="1" y="1" width="36" height="20" rx="10" fill="#252E47" stroke="#475569" stroke-width="1.5"/>
  <circle cx="11" cy="11" r="7" fill="#94A3B8"/>
</svg>"""
    (assets_dir / "toggle_off_hover.svg").write_text(toggle_off_hover_svg, encoding="utf-8")

    # Toggle On (filled with accent)
    toggle_on_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="38" height="22" viewBox="0 0 38 22">
  <rect x="1" y="1" width="36" height="20" rx="10" fill="{accent}" stroke="{accent}" stroke-width="1.5"/>
  <circle cx="27" cy="11" r="7" fill="#FFFFFF"/>
</svg>"""
    (assets_dir / "toggle_on.svg").write_text(toggle_on_svg, encoding="utf-8")

    # Toggle On Hover
    toggle_on_hover_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="38" height="22" viewBox="0 0 38 22">
  <rect x="1" y="1" width="36" height="20" rx="10" fill="{accent}" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="27" cy="11" r="7" fill="#FFFFFF"/>
</svg>"""
    (assets_dir / "toggle_on_hover.svg").write_text(toggle_on_hover_svg, encoding="utf-8")

    # Dropdown Chevron
    chevron_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">
  <path d="M2.5 4.5L6 8L9.5 4.5" stroke="#94A3B8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    (assets_dir / "chevron_down.svg").write_text(chevron_svg, encoding="utf-8")


def get_stylesheet(accent: str = ACCENT_COLOR) -> str:
    """
    Constructs and returns the full Qt Stylesheet string.
    """
    assets_dir = Path(__file__).resolve().parent / "assets"
    _generate_assets(accent, assets_dir)
    assets_posix = assets_dir.as_posix()

    return f"""
/* ====================================================================
   GLOBAL & WINDOW STYLING
   ==================================================================== */
QMainWindow, QWidget#central_widget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
}}

* {{
    font-family: "Segoe UI Variable", "Segoe UI Variable Text", "Inter", "Segoe UI", -apple-system, sans-serif;
    outline: none;
}}

/* ====================================================================
   ROUNDED CARDS
   ==================================================================== */
QFrame#card_camera, 
QFrame#card_transcription, 
QFrame#card_settings {{
    background-color: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
}}

/* Top App Header */
QFrame#header_bar {{
    background-color: {BG_DARK};
    border-bottom: 1px solid #1B2133;
    padding: 2px 4px;
}}

QLabel#app_title {{
    color: {TEXT_PRIMARY};
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

QLabel#app_badge {{
    background-color: rgba(108, 99, 255, 0.15);
    color: {accent};
    border: 1px solid rgba(108, 99, 255, 0.35);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}}

QLabel#app_subtitle {{
    color: {TEXT_MUTED};
    font-size: 12px;
    font-weight: 400;
}}

QLabel#status_pill {{
    background-color: rgba(16, 185, 129, 0.12);
    color: {SUCCESS_GREEN};
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
}}

/* Section Headers */
QLabel[section="true"] {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    margin-top: 8px;
    margin-bottom: 4px;
    min-height: 16px;
}}

/* ====================================================================
   CAMERA FEED & PREDICTION BADGE
   ==================================================================== */
QLabel#video_label {{
    background-color: {INPUT_BG};
    border: 1px solid #1E2538;
    border-radius: 12px;
    color: {TEXT_SECONDARY};
}}

QLabel#pred_badge {{
    background-color: rgba(108, 99, 255, 0.12);
    border: 1.5px solid {accent};
    border-radius: 10px;
    color: {TEXT_PRIMARY};
    font-size: 16px;
    font-weight: 700;
    padding: 8px 18px;
    letter-spacing: 0.5px;
}}

QLabel#warning_label {{
    color: #F59E0B;
    font-size: 12px;
    font-weight: 500;
    padding: 2px;
}}

/* ====================================================================
   TEXT BOXES
   ==================================================================== */
QTextEdit {{
    background-color: {INPUT_BG};
    border: 1px solid {INPUT_BORDER};
    border-radius: 10px;
    color: #F8FAFC;
    padding: 10px 12px;
    selection-background-color: {accent};
    selection-color: #FFFFFF;
}}

QTextEdit:focus {{
    border: 1.5px solid {accent};
    background-color: #0D121F;
}}

QTextEdit#translation_text {{
    color: #34D399;
    font-size: 15px;
    font-weight: 600;
}}

/* ====================================================================
   BUTTONS
   ==================================================================== */
/* Primary Buttons (Start Camera, Translate) */
QPushButton#btn_camera,
QPushButton#btn_translate {{
    background-color: {accent};
    color: #FFFFFF;
    border: 1px solid {accent};
    border-radius: 9px;
    padding: 9px 16px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton#btn_camera:hover,
QPushButton#btn_translate:hover {{
    background-color: #7B73FF;
    border-color: #7B73FF;
}}

QPushButton#btn_camera:pressed,
QPushButton#btn_translate:pressed {{
    background-color: #584FE4;
    border-color: #584FE4;
}}

QPushButton#btn_camera:disabled,
QPushButton#btn_translate:disabled {{
    background-color: #232B42;
    border-color: #232B42;
    color: {TEXT_MUTED};
}}

/* Red Stop Style for Camera Button */
QPushButton#btn_camera[running="true"],
QPushButton#btn_camera[text*="Stop"] {{
    background-color: {DANGER_RED};
    border: 1px solid {DANGER_RED};
    color: #FFFFFF;
}}

QPushButton#btn_camera[running="true"]:hover,
QPushButton#btn_camera[text*="Stop"]:hover {{
    background-color: {DANGER_RED_HOVER};
    border-color: {DANGER_RED_HOVER};
}}

QPushButton#btn_camera[running="true"]:pressed,
QPushButton#btn_camera[text*="Stop"]:pressed {{
    background-color: {DANGER_RED_PRESSED};
    border-color: {DANGER_RED_PRESSED};
}}

/* Secondary Outlined Buttons */
QPushButton#btn_space,
QPushButton#btn_delete,
QPushButton#btn_clear,
QPushButton#btn_speak {{
    background-color: #121624;
    color: #E2E8F0;
    border: 1px solid #2B354F;
    border-radius: 9px;
    padding: 0px 10px;
    min-height: 36px;
    font-size: 12px;
    font-weight: 600;
}}

QPushButton#btn_space:hover,
QPushButton#btn_speak:hover {{
    background-color: #1E253A;
    border-color: #485675;
    color: #FFFFFF;
}}

QPushButton#btn_space:pressed,
QPushButton#btn_speak:pressed {{
    background-color: #0E1320;
}}

QPushButton#btn_space:disabled,
QPushButton#btn_speak:disabled,
QPushButton#btn_delete:disabled,
QPushButton#btn_clear:disabled {{
    background-color: #141926;
    border-color: #202738;
    color: #475569;
}}

/* Red Hover on Delete & Clear */
QPushButton#btn_delete:hover,
QPushButton#btn_clear:hover {{
    background-color: rgba(239, 68, 68, 0.12);
    border-color: {DANGER_RED};
    color: {DANGER_RED};
}}

QPushButton#btn_delete:pressed,
QPushButton#btn_clear:pressed {{
    background-color: rgba(239, 68, 68, 0.22);
    border-color: {DANGER_RED_HOVER};
    color: #FCA5A5;
}}

/* ====================================================================
   SEGMENTED CONTROL (STATIC / DYNAMIC MODE)
   ==================================================================== */
QFrame#segmented_control {{
    background-color: {INPUT_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 3px;
    min-height: 46px;
}}

QFrame#segmented_control QPushButton {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-radius: 7px;
    padding: 0px 14px;
    min-height: 40px;
    max-height: 40px;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
}}

QFrame#segmented_control QPushButton:hover {{
    color: #FFFFFF;
    background-color: rgba(255, 255, 255, 0.05);
}}

QFrame#segmented_control QPushButton:checked {{
    background-color: {accent};
    color: #FFFFFF;
    font-weight: 700;
}}

QFrame#segmented_control QPushButton:checked:hover {{
    background-color: #7B73FF;
}}

/* Dynamic Vocab pill */
QLabel#vocab_label {{
    color: {TEXT_MUTED};
    font-size: 11px;
    background-color: #111624;
    border: 1px solid #1E2538;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 22px;
}}

/* ====================================================================
   DROPDOWN (COMBOBOX)
   ==================================================================== */
QComboBox {{
    background-color: {INPUT_BG};
    border: 1px solid {INPUT_BORDER};
    border-radius: 8px;
    padding: 0px 14px;
    color: #E2E8F0;
    font-size: 13px;
    min-height: 40px;
    max-height: 40px;
}}

QComboBox:hover, 
QComboBox:focus {{
    border: 1px solid {accent};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: right center;
    width: 32px;
    height: 40px;
    border-left: none;
}}

QComboBox::down-arrow {{
    image: url("{assets_posix}/chevron_down.svg");
    width: 14px;
    height: 14px;
}}

QComboBox QAbstractItemView {{
    background-color: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    color: #E2E8F0;
    selection-background-color: {accent};
    selection-color: #FFFFFF;
    padding: 4px;
    outline: none;
}}

/* ====================================================================
   CHECKBOXES AS TOGGLE SWITCHES
   ==================================================================== */
QCheckBox {{
    color: #E2E8F0;
    font-size: 13px;
    font-weight: 500;
    spacing: 12px;
    min-height: 32px;
}}

QCheckBox:hover {{
    color: #FFFFFF;
}}

QCheckBox::indicator {{
    subcontrol-origin: padding;
    subcontrol-position: left center;
    width: 38px;
    height: 22px;
}}

QCheckBox::indicator:unchecked {{
    image: url("{assets_posix}/toggle_off.svg");
}}

QCheckBox::indicator:unchecked:hover {{
    image: url("{assets_posix}/toggle_off_hover.svg");
}}

QCheckBox::indicator:checked {{
    image: url("{assets_posix}/toggle_on.svg");
}}

QCheckBox::indicator:checked:hover {{
    image: url("{assets_posix}/toggle_on_hover.svg");
}}

/* ====================================================================
   THIN SLIDERS WITH ACCENT TRACK & ROUND HANDLE
   ==================================================================== */
QSlider::groove:horizontal {{
    height: 5px;
    background: #232B42;
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}

QSlider::add-page:horizontal {{
    background: #1B2133;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid {accent};
    width: 15px;
    height: 15px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: {accent};
    border: 2px solid #FFFFFF;
}}

QLabel.slider_val {{
    color: {accent};
    font-weight: 700;
    font-size: 12px;
    background-color: rgba(108, 99, 255, 0.1);
    border: 1px solid rgba(108, 99, 255, 0.25);
    border-radius: 6px;
    padding: 2px 7px;
    min-width: 34px;
}}

/* ====================================================================
   SCROLL AREA & SCROLLBAR (PREVENTS SQUEEZING ON SMALL SCREENS)
   ==================================================================== */
QScrollArea#right_scroll_area {{
    background-color: transparent;
    border: none;
}}

QWidget#right_scroll_content {{
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {BG_DARK};
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {CARD_BORDER};
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {accent};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: none;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* ====================================================================
   STATUS BAR & TOOLTIPS
   ==================================================================== */
QStatusBar {{
    background-color: {INPUT_BG};
    border-top: 1px solid #1E2538;
    color: {TEXT_MUTED};
    font-size: 11px;
    padding: 3px 12px;
}}

QToolTip {{
    background-color: #1A2133;
    color: #FFFFFF;
    border: 1px solid #2E3850;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
"""

