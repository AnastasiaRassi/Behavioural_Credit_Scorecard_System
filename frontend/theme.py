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
/* Shorter run-in at the top, and a readable measure. Full-width text on a 1440px
   monitor puts related numbers too far apart to compare. */
[data-testid="stMainBlockContainer"] {{
    padding-top: 3rem;
    padding-bottom: 5rem;
    max-width: 1180px;
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

[data-testid="stCaptionContainer"] p {{ color: {MUTED}; line-height: 1.5; }}
hr {{ border-color: {GRID} !important; opacity: 1; }}
[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
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
