"""One palette, one chart theme, applied once at startup.

Colour carries meaning on this dashboard. Green, amber and red mean risk band and
audit outcome; blue is the neutral series colour and the interface accent. Keeping
the accent out of the traffic-light range stops a button or an active tab from
reading as a verdict.

The same four values were previously repeated as literals across eight modules,
so a chart could drift away from the band colour it was meant to match. Import
from here instead.
"""
import altair as alt
import streamlit as st

# Risk bands and audit outcomes. The only three colours allowed to mean something.
LOW = "#35B36A"
MEDIUM = "#D97706"
HIGH = "#DC2626"

# Neutral series and interface accent. Mirrors primaryColor in .streamlit/config.toml.
ACCENT = "#4C8DF6"

# Loana AI's card. Not a risk colour: it only ever fills the assistant's column.
LOANA = "#00C588"
LOANA_INK = "#073B2D"

# Cardholder timeline: neon mint for what was paid, slate for what was billed.
PAID = "#00E699"
BILLED = "#64748B"

BAND_COLOURS ={"Low": LOW, "Medium": MEDIUM, "High": HIGH}
BAND_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# Chart furniture: espresso text and warm stone on white cards.
TEXT = "#1C1917"
MUTED = "#78716C"
GRID = "#E6E2DE"
SURFACE = "#FFFFFF"

FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Helvetica Neue", '
    "Arial, sans-serif"
)

_CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");
/* Inter everywhere: banner, Loana and the rest of the app. Icon fonts are kept. */
html, body, [data-testid="stApp"], .ct-banner,
[data-testid="stApp"] :is(p, span, div, label, button, input, textarea, h1, h2, h3, h4, li, td, th):not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
    font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
}}
/* Warm ecru page; the header is transparent so it shows through. */
[data-testid="stApp"] {{ background: #FBF9F6; color: {TEXT}; }}
[data-testid="stApp"] :is(h1, h2, h3, h4, p, li, label) {{ color: {TEXT}; }}
[data-testid="stCaptionContainer"] p {{ color: {MUTED}; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* Loana AI (ui/loana.py): a fixed bubble, or a fixed 380x550 card, bottom-right. */
.st-key-loana_launch {{
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    z-index: 1000;
    width: auto;
}}
/* A pill: the invitation on the left, the avatar (a CSS background, so the button
   stays a real button) on the right. */
.st-key-loana_launch button {{
    width: auto;
    height: 112px;
    min-height: 0;
    padding: 0 132px 0 1.6rem;
    border: 3px solid {LOANA};
    border-radius: 999px;
    background-color: #FFFFFF;
    background-size: auto 100%;
    background-repeat: no-repeat;
    background-position: right 8px center;
    box-shadow: 0 8px 22px rgba(7, 59, 45, 0.3);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    animation: loana-nudge 2.6s ease-in-out 1.5s 3;
}}
.st-key-loana_launch button p {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #1C1917;
    white-space: nowrap;
}}
.st-key-loana_launch button:hover {{
    transform: scale(1.05);
    border-color: {LOANA_INK};
    box-shadow: 0 12px 28px rgba(7, 59, 45, 0.42);
}}
@keyframes loana-nudge {{
    0%, 100% {{ transform: none; }}
    50% {{ transform: translateY(-6px); }}
}}
@keyframes loana-pop {{
    from {{ opacity: 0; transform: translateY(16px) scale(0.96); }}
    to {{ opacity: 1; transform: none; }}
}}
.st-key-loana_panel {{
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    z-index: 1000;
    width: min(460px, calc(100vw - 2rem));
    height: min(760px, calc(100vh - 3rem));
    gap: 0.6rem;
    padding: 1.1rem;
    overflow-y: auto;
    background: {LOANA};
    border-radius: 22px;
    box-shadow: 0 16px 44px rgba(7, 59, 45, 0.35);
    animation: loana-pop 0.22s ease-out;
}}
.st-key-loana_close {{
    position: absolute;
    top: 0.7rem;
    right: 0.7rem;
    z-index: 2;
    width: auto;
}}
.st-key-loana_panel .st-key-loana_close button {{
    width: 32px;
    height: 32px;
    min-height: 0;
    padding: 0;
    background: rgba(255, 255, 255, 0.85);
}}
.st-key-loana_panel > * {{ flex-shrink: 0; }}
/* The messages scroll inside the card; everything under them (presets, input)
   keeps its place at the bottom. */
[data-testid="stLayoutWrapper"]:has(> .st-key-loana_chat) {{
    flex: 1 1 0;
    min-height: 200px;
}}
.st-key-loana_chat {{ height: 100% !important; }}
/* The messages fade out at the bottom edge of their fixed-height box. The padding
   is the fade's own height, so the last message can still scroll clear of it. */
.st-key-loana_chat {{
    padding-bottom: 32px;
    -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - 32px), transparent);
    mask-image: linear-gradient(to bottom, #000 calc(100% - 32px), transparent);
}}

.loana-head {{ display: flex; align-items: center; gap: 0.85rem; }}
.loana-head img {{
    width: 88px;
    height: 88px;
    flex: none;
    padding: 4px;
    background: #FFFFFF;
    border-radius: 50%;
    box-shadow: 0 4px 14px rgba(7, 59, 45, 0.25);
}}
/* Padding, not margin: Streamlit trims a markdown block's last 1rem. */
.loana-head {{ min-width: 0; padding-bottom: 1rem; }}
.loana-head .loana-name {{
    font-size: clamp(1.1rem, 1.9vw, 1.4rem);
    font-weight: 600;
    letter-spacing: 0.5px;
    line-height: 1.1;
    white-space: nowrap;
    color: #0F172A;
}}
.loana-head .loana-sub {{
    margin-top: 0.15rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: rgba(15, 23, 42, 0.78);
}}

.st-key-loana_panel p, .st-key-officer_triage p {{ color: {LOANA_INK}; }}
.st-key-loana_panel [data-testid="stCaptionContainer"] p,
.st-key-officer_triage [data-testid="stCaptionContainer"] p {{
    color: rgba(7, 59, 45, 0.78);
}}
.st-key-loana_panel [data-testid="stChatMessage"],
.st-key-officer_triage [data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.92);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    align-items: flex-start;
}}
.st-key-loana_panel [data-testid="stChatMessage"] p,
.st-key-officer_triage [data-testid="stChatMessage"] p {{
    line-height: 1.55;
    text-align: left;
}}
.st-key-loana_panel [data-testid="stChatMessage"] p,
.st-key-officer_triage [data-testid="stChatMessage"] p {{ color: #1D2B27; }}
.st-key-loana_panel [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p,
.st-key-officer_triage [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {{
    color: {MUTED};
    text-align: left;
}}
.st-key-loana_panel button,
.st-key-officer_triage button {{
    background: #FFFFFF;
    border: none;
    color: {LOANA_INK};
    border-radius: 999px;
}}
.st-key-loana_panel button:hover,
.st-key-officer_triage button:hover {{ background: #E9FFF7; color: {LOANA_INK}; }}

/* Cardholder timeline card (views/cardholder.py). */
.st-key-history_card {{
    background: #FFFFFF;
    border-radius: 18px;
    box-shadow: 0 6px 24px rgba(28, 25, 23, 0.06);
    padding: 0.6rem 0.9rem;
}}
.history-head {{ display: flex; align-items: center; gap: 0.9rem; }}
.history-icon {{
    flex: none;
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font: italic 700 1.35rem Georgia, serif;
    color: #00754F;
    background: #E3FFF4;
    border: 2px solid {PAID};
    box-shadow: 0 0 12px rgba(0, 230, 153, 0.45);
}}
.history-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }}
.history-hint {{ font-size: 0.95rem; color: {MUTED}; }}
.loana-hint {{
    margin: 0.4rem 0 1.2rem;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: #F0FFF9;
    border-left: 4px solid {LOANA};
    color: {LOANA_INK};
}}

/* Shorter run-in at the top. The page uses the full width: the cardholder screen is
   two columns, and a 1180px cap left the copilot column too narrow to read. The
   sign-in page sets its own width (ui/login.css). */
[data-testid="stMainBlockContainer"] {{
    padding-top: 3rem;
    padding-bottom: 2rem;
    padding-left: clamp(1.5rem, 4vw, 4rem);
    padding-right: clamp(1.5rem, 4vw, 4rem);
    max-width: none;
}}

/* The page title competes with the metrics underneath it at Streamlit's default
   size, and this screen is about the numbers. */
h1 {{ font-size: 2.1rem !important; }}
h2 {{ font-size: 1.5rem !important; }}
h3 {{ font-size: 1.25rem !important; }}
/* One type scale for every screen: bold tight headings, medium muted subtext,
   semibold labels. Streamlit's own heading and caption styles are weaker. */
[data-testid="stApp"] {{
    line-height: 1.55;
    letter-spacing: -0.003em;
    -webkit-font-smoothing: antialiased;
}}
[data-testid="stApp"] :is(h1, h2, h3, h4) {{
    font-weight: 700;
    line-height: 1.25;
    letter-spacing: -0.02em;
    color: {TEXT};
}}
[data-testid="stHeading"] {{ margin-bottom: 0.15rem; }}
[data-testid="stCaptionContainer"] {{ opacity: 1; }}
[data-testid="stCaptionContainer"] p {{
    font-size: 0.9rem;
    font-weight: 500;
    line-height: 1.5;
}}
[data-testid="stWidgetLabel"] p {{
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0;
    color: {TEXT};
}}
[data-testid="stMarkdownContainer"] strong {{ font-weight: 600; }}

/* Metric tiles: the number is the content, the label is the furniture. */
[data-testid="stMetricValue"] {{
    font-size: 2.3rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.78rem !important;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
}}

/* The 💭 triggers are an aid, not an action. As default secondary buttons there
   was a row of grey pills under every metric, louder than the numbers. */
[data-testid="stPopoverButton"] {{
    background: transparent !important;
    border: 1px solid {GRID} !important;
    color: {MUTED} !important;
    padding: 0.1rem 0.5rem !important;
    min-height: 0 !important;
    border-radius: 999px !important;
}}
[data-testid="stPopoverButton"]:hover {{
    border-color: {ACCENT} !important;
    color: {TEXT} !important;
}}

/* Tabs read as navigation rather than as links in a paragraph. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
[data-testid="stTab"] {{
    font-weight: 600;
    letter-spacing: 0.01em;
    padding-left: 0.85rem;
    padding-right: 0.85rem;
}}

/* Bordered containers become cards rather than boxes drawn on the background. */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background: {SURFACE};
    border: 1px solid {GRID};
    border-radius: 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.04);
}}
[data-testid="stMetric"] {{ padding: 0.9rem 1.1rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0.35rem 0.5rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
}}
/* Only the review-request cards lift: they are the ones with a Review button. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.officer-card-head--lift):hover {{
    transform: translateY(-2px);
    border-color: #D6D0CA;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04), 0 10px 24px rgba(0, 0, 0, 0.08);
}}
/* The Loana card and its chat are not white cards. */
.st-key-loana_panel [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent; border: none; box-shadow: none;
}}

/* Officer cards (ui/components.py). Red, amber and green appear only on the band
   badge and the reason arrows, where they mean risk. */
.officer-card-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
}}
.officer-card-title {{
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    line-height: 1.3;
}}
.officer-card-meta {{ font-size: 0.85rem; font-weight: 500; color: {MUTED}; margin-top: 0.15rem; }}
.officer-badge {{
    flex: none;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    border: 1px solid {MUTED};
    white-space: nowrap;
}}
.officer-badge--Low {{ border-color: {LOW}; }}
.officer-badge--Medium {{ border-color: {MEDIUM}; }}
.officer-badge--High {{ border-color: {HIGH}; }}
.officer-quote {{
    margin: 0.25rem 0 0;
    padding-left: 0.8rem;
    border-left: 3px solid {GRID};
    line-height: 1.5;
}}
.officer-arrow--up {{ color: {HIGH}; }}
.officer-arrow--down {{ color: {LOW}; }}

[data-testid="stCaptionContainer"] p {{ color: {MUTED}; line-height: 1.5; }}
hr {{ border-color: {GRID} !important; opacity: 1; }}
[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

/* The officer's search is the way in, so it is the biggest thing on the page.
   The border lives on Streamlit's root wrapper, not the <input>, so focus is
   styled on the wrapper too. */
.st-key-officer_lookup [data-testid="stTextInputRootElement"] {{
    background: #F5F2EE;
    border: 1px solid {GRID};
    border-radius: 12px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}}
.st-key-officer_lookup [data-testid="stTextInputRootElement"]:hover {{ border-color: #CFC8C1; }}
.st-key-officer_lookup [data-testid="stTextInputRootElement"]:focus-within {{
    background: #FFFFFF;
    border-color: {ACCENT};
    box-shadow: 0 0 0 3px rgba(76, 141, 246, 0.18);
}}
.st-key-officer_lookup input {{
    font-size: 1.05rem;
    font-weight: 500;
    padding: 0.85rem 1.1rem;
    background: transparent;
}}
.st-key-officer_lookup input::placeholder {{ color: {MUTED}; font-weight: 400; }}

/* Landing hero: room between the sections, and the illustration as a soft card. */
[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) {{ gap: 3.5rem; }}
[data-testid="stImage"] img {{
    border-radius: 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.04);
}}
[data-testid="stMainBlockContainer"] hr {{ margin: 1.75rem 0; }}

/* Officer landing: the search box is compact, and Loana's card sticks under the
   page bar. Sticky needs the column to span the row, hence align-items: stretch. */
.st-key-officer_lookup input {{ padding: 0.55rem 1rem; font-size: 1rem; }}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-officer_triage) {{
    align-items: stretch;
}}
.st-key-officer_triage {{
    position: sticky;
    top: 7.5rem;
    max-height: calc(100vh - 8.5rem);
    overflow-y: auto;
}}
/* The triage panel is Loana's card, the same emerald as the floating chat. */
.st-key-officer_triage {{
    background: {LOANA};
    border: none;
    border-radius: 22px;
    padding: 1.1rem;
    box-shadow: 0 16px 44px rgba(7, 59, 45, 0.35);
}}
</style>
"""


@alt.theme.register("credit_risk", enable=True)
def _chart_theme() -> alt.theme.ThemeConfig:
    """Shared chart styling, so no tab has to restate fonts, grid or axis colour."""
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": "transparent",
                "font": FONT,
                "view": {"stroke": "transparent", "continuousHeight": 300},
                "axis": {
                    "labelColor": MUTED,
                    "titleColor": MUTED,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "titleFontWeight": 600,
                    "titlePadding": 10,
                    "gridColor": GRID,
                    "domainColor": GRID,
                    "tickColor": GRID,
                },
                "legend": {
                    "labelColor": TEXT,
                    "titleColor": MUTED,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "symbolType": "square",
                },
                "title": {"color": TEXT, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
                "range": {"category": [ACCENT, LOW, MEDIUM, HIGH, MUTED]},
                "bar": {"color": ACCENT, "cornerRadiusEnd": 3},
                "line": {"color": ACCENT, "strokeWidth": 2},
                "point": {"color": ACCENT, "filled": True},
                "rule": {"color": MUTED},
            }
        }
    )


def apply() -> None:
    """Inject the stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
