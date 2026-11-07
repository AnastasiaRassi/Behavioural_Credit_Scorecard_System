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
MEDIUM = "#E0A92B"
HIGH = "#E05545"

# Neutral series and interface accent. Mirrors primaryColor in .streamlit/config.toml.
ACCENT = "#4C8DF6"

# Loana AI's card. Not a risk colour: it only ever fills the assistant's column.
LOANA = "#00C588"
LOANA_INK = "#073B2D"

BAND_COLOURS = {"Low": LOW, "Medium": MEDIUM, "High": HIGH}
BAND_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# Chart furniture, dark-theme matched.
TEXT = "#E4E9F2"
MUTED = "#8C98AC"
GRID = "#222C3C"
SURFACE = "#172030"

FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Helvetica Neue", '
    "Arial, sans-serif"
)

_CSS = f"""
<style>
/* One light-blue gradient behind every signed-in page; the header is transparent
   so it shows through. */
[data-testid="stApp"] {{
    background: linear-gradient(160deg, #EDF6FF 0%, #D9EBFC 50%, #C5DEF6 100%) fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Loana AI (ui/loana.py). The column stays transparent and stretches to the row;
   the card inside it is sticky and one screen tall, so it follows the page instead
   of leaving green blank below a short chat. Streamlit gives columns and layout
   wrappers no key, hence :has(). */
[data-testid="stColumn"]:has(.st-key-loana_panel) {{
    align-self: stretch;
    display: flex;
    flex-direction: column;
}}
[data-testid="stColumn"]:has(.st-key-loana_panel) > [data-testid="stVerticalBlock"] {{
    flex: 1 1 auto;
}}
[data-testid="stLayoutWrapper"]:has(> .st-key-loana_panel) {{
    position: sticky;
    top: 4rem;
    height: calc(100vh - 6.5rem);
}}
.st-key-loana_panel {{
    height: 100%;
    gap: 0.6rem;
    padding: 1.1rem;
    overflow-y: auto;
    background: {LOANA};
    border-radius: 22px;
    box-shadow: 0 12px 32px rgba(0, 197, 136, 0.28);
}}
.st-key-loana_panel > * {{ flex-shrink: 0; }}
/* The messages scroll inside the card; everything under them (presets, input)
   keeps its place at the bottom. */
[data-testid="stLayoutWrapper"]:has(> .st-key-loana_chat) {{
    flex: 1 1 0;
    min-height: 200px;
}}
.st-key-loana_chat {{ height: 100% !important; }}
@media (max-width: 640px) {{
    [data-testid="stLayoutWrapper"]:has(> .st-key-loana_panel) {{
        position: static;
        height: auto;
    }}
    [data-testid="stLayoutWrapper"]:has(> .st-key-loana_chat) {{ flex: none; }}
    .st-key-loana_chat {{ height: 420px !important; }}
}}

.loana-head {{ display: flex; align-items: center; gap: 0.85rem; }}
.loana-head img {{
    width: 64px;
    height: 64px;
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
    font-weight: 800;
    letter-spacing: 0.14em;
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

.st-key-loana_panel p {{ color: {LOANA_INK}; }}
.st-key-loana_panel [data-testid="stCaptionContainer"] p {{
    color: rgba(7, 59, 45, 0.78);
}}
.st-key-loana_panel [data-testid="stChatMessage"] {{
    background: rgba(255, 255, 255, 0.92);
    border-radius: 14px;
}}
.st-key-loana_panel [data-testid="stChatMessage"] p {{ color: #1D2B27; }}
.st-key-loana_panel [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {{
    color: {MUTED};
    text-align: left;
}}
.st-key-loana_panel button {{
    background: #FFFFFF;
    border: none;
    color: {LOANA_INK};
    border-radius: 999px;
}}
.st-key-loana_panel button:hover {{ background: #E9FFF7; color: {LOANA_INK}; }}

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
h1 {{ font-size: 2.1rem !important; letter-spacing: -0.02em; }}
h2 {{ font-size: 1.45rem !important; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.12rem !important; }}

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
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE};
    border-radius: 10px;
}}

/* Officer cards (ui/components.py). Red, amber and green appear only on the band
   badge and the reason arrows, where they mean risk. */
.officer-card-head {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
}}
.officer-card-title {{ font-size: 1.05rem; font-weight: 650; line-height: 1.3; }}
.officer-card-meta {{ font-size: 0.85rem; color: {MUTED}; margin-top: 0.1rem; }}
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

/* The officer's search is the way in, so it is the biggest thing on the page. */
.st-key-officer_lookup input {{
    font-size: 1.15rem;
    padding: 0.9rem 1.1rem;
    background: #FFFFFF;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(41, 98, 160, 0.14);
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
