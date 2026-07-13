"""Shared visual styling for every Streamlit page: theme CSS, brand logo,
page header, step indicator, and status badges.

Palette follows the uWallet brand mark (near-black navy fading into red,
white wordmark), applied as accents -- sidebar and hero banner -- on a
clean white page background. Kept purely presentational -- no business
logic lives here. Every page calls apply_theme() once near the top, then
uses render_header() / render_stepper() / status_badge() instead of raw
st.title()/st.metric() calls, so the whole app looks like one consistent
branded product instead of five separate default-Streamlit pages.
"""

import streamlit as st

STEPS = [
    ("1", "Upload", "📤"),
    ("2", "Process", "⚙️"),
    ("3", "Review", "🧾"),
    ("4", "Export", "📦"),
]

# Brand accent colors (from the uWallet mark), used for the sidebar/hero/
# buttons. The page background itself is white -- see PAGE_BG below.
BRAND_NAVY = "#0B1226"
BRAND_NAVY_LIGHT = "#141B33"
BRAND_RED = "#E63946"
BRAND_RED_DARK = "#B3202E"
BRAND_TEXT = "#F5F7FB"

# Main page palette: white background, dark text, light card surfaces.
PAGE_BG = "#FFFFFF"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E7E3E6"
TEXT_DARK = "#1B2430"
TEXT_MUTED = "#6B7284"

# Badge background/foreground pairs tuned for light card surfaces.
_BADGE_COLORS = {
    "success": ("#E6F4EA", "#1E7A34"),
    "warning": ("#FFF4E0", "#9A6400"),
    "error": ("#FDEBEC", "#B3261E"),
    "neutral": ("#F0F1F5", "#3B4758"),
    "info": ("#E8F0FE", "#1A56C4"),
}

# Inline SVG recreation of the uWallet mark: navy-to-red diagonal gradient
# tile with the white "uwallet" wordmark, so it renders identically in
# Docker/local without needing a separate static-asset file to wire up.
_LOGO_SVG = """
<svg width="{width}" height="{height}" viewBox="0 0 220 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="uWallet">
    <defs>
        <linearGradient id="uwGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0B1226"/>
            <stop offset="55%" stop-color="#4A1420"/>
            <stop offset="100%" stop-color="#E63946"/>
        </linearGradient>
    </defs>
    <rect width="220" height="64" rx="14" fill="url(#uwGrad)"/>
    <text x="20" y="41" font-family="Segoe UI, Arial, sans-serif" font-size="28"
          font-weight="700" fill="#FFFFFF" letter-spacing="0.5">uwallet</text>
</svg>
"""


def render_logo(width: int = 160, height: int = 46) -> str:
    """Return the inline-SVG uWallet logo markup, sized for the caller."""
    return _LOGO_SVG.format(width=width, height=height)


def apply_theme() -> None:
    """Inject the shared CSS. Safe to call on every page (idempotent)."""
    st.markdown(
        f"""
        <style>
        /* ---- App shell ------------------------------------------------ */
        .stApp {{
            background: {PAGE_BG};
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }}

        /* ---- Sidebar (kept as a dark brand accent on the white page) ------ */
        [data-testid="stSidebar"] {{
            background: linear-gradient(160deg, {BRAND_NAVY} 0%, #201029 55%, {BRAND_RED_DARK} 130%);
            border-right: 1px solid rgba(230, 57, 70, 0.25);
        }}
        [data-testid="stSidebar"] * {{
            color: {BRAND_TEXT} !important;
        }}
        [data-testid="stSidebar"] .stAlert {{
            background-color: rgba(255,255,255,0.08);
            border-radius: 10px;
        }}
        .sidebar-logo {{
            margin-bottom: 0.6rem;
        }}

        /* ---- Headings / body text (main content area is white) ------------- */
        .main h1, .main h2, .main h3, .main h4,
        .main p, .main span, .main label, .main .stMarkdown {{
            color: {TEXT_DARK};
        }}
        .main h1, .main h2, .main h3 {{
            font-weight: 700;
        }}

        /* ---- Buttons ------------------------------------------------------ */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 10px;
            border: none;
            background: linear-gradient(135deg, {BRAND_RED} 0%, {BRAND_RED_DARK} 100%);
            color: #FFFFFF;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(230, 57, 70, 0.25);
            transition: transform 0.05s ease-in-out;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(230, 57, 70, 0.4);
            color: #FFFFFF;
        }}

        /* ---- Metrics as cards ---------------------------------------------- */
        [data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 0.9rem 1rem 0.6rem 1rem;
            box-shadow: 0 1px 4px rgba(20, 20, 30, 0.06);
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT_DARK} !important;
        }}

        /* ---- Alerts / expanders / file uploader --------------------------- */
        .stAlert {{
            border-radius: 12px;
        }}
        [data-testid="stExpander"] {{
            border-radius: 12px;
            border: 1px solid {CARD_BORDER};
            background: {CARD_BG};
        }}
        [data-testid="stFileUploaderDropzone"] {{
            border-radius: 14px;
            background: #FBFBFC;
            border: 1px dashed #D9D5D8;
        }}
        [data-testid="stExpanderDetails"] pre {{
            background: #0F1B33;
            color: #D6E2FF;
            border-radius: 10px;
        }}

        /* ---- Bordered containers (used for cards) -------------------------- */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 16px !important;
            border-color: {CARD_BORDER} !important;
            background: {CARD_BG};
        }}

        /* ---- Hero header ---------------------------------------------------- */
        .hero {{
            background: linear-gradient(135deg, {BRAND_NAVY} 0%, #4A1420 55%, {BRAND_RED} 100%);
            border-radius: 18px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 6px 20px rgba(27, 18, 20, 0.22);
        }}
        .hero h1 {{
            color: #FFFFFF !important;
            margin: 0 0 0.35rem 0;
            font-size: 1.7rem;
        }}
        .hero p {{
            color: #EBD9DD;
            margin: 0;
            font-size: 0.98rem;
        }}

        /* ---- Step indicator --------------------------------------------------- */
        .stepper {{
            display: flex;
            align-items: center;
            margin-bottom: 1.8rem;
        }}
        .step {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .step-circle {{
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            flex-shrink: 0;
        }}
        .step-label {{
            font-weight: 600;
            font-size: 0.92rem;
            white-space: nowrap;
        }}
        .step-line {{
            flex: 1;
            height: 3px;
            margin: 0 0.6rem;
            border-radius: 2px;
        }}

        /* ---- Badges ------------------------------------------------------------ */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Consistent sidebar branding (logo + tagline), called once per page."""
    st.sidebar.markdown(
        f'<div class="sidebar-logo">{render_logo(150, 44)}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Court Order Extraction · Local OCR + Rule-based Extraction + Arabic NER")
    st.sidebar.info("No LLM · No Cloud AI · No External API")


def render_header(icon: str, title: str, subtitle: str) -> None:
    """Gradient hero banner used at the top of every page instead of st.title()."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current_step: int) -> None:
    """Horizontal 1-4 step indicator (Upload / Process / Review / Export)."""
    parts = ['<div class="stepper">']

    for index, (number, label, icon) in enumerate(STEPS, start=1):
        if index < current_step:
            circle_bg, circle_fg, label_color = BRAND_RED, "#FFFFFF", BRAND_RED
            content = "✓"
        elif index == current_step:
            circle_bg, circle_fg, label_color = BRAND_NAVY, "#FFFFFF", TEXT_DARK
            content = icon
        else:
            circle_bg, circle_fg, label_color = "#EDEBEC", "#9AA0AC", TEXT_MUTED
            content = number

        parts.append(
            f"""
            <div class="step">
                <div class="step-circle" style="background:{circle_bg}; color:{circle_fg}; border:1px solid rgba(230,57,70,0.25);">{content}</div>
                <div class="step-label" style="color:{label_color};">{label}</div>
            </div>
            """
        )

        if index != len(STEPS):
            line_color = BRAND_RED if index < current_step else "#EDEBEC"
            parts.append(f'<div class="step-line" style="background:{line_color};"></div>')

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def status_badge(text: str, tone: str = "neutral") -> str:
    """Return an inline HTML pill badge. Caller wraps it in st.markdown(..., unsafe_allow_html=True)."""
    bg, fg = _BADGE_COLORS.get(tone, _BADGE_COLORS["neutral"])
    return f'<span class="badge" style="background:{bg}; color:{fg};">{text}</span>'
