"""Small HTML pieces shared by the signed-in screens.

Each returns one line of markup: Streamlit renders through Markdown, and an
indented line after a blank one becomes a code fence.
"""
import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

from money import html_safe
from theme import BAND_ICONS

# The two directions a reason can pull. Not the risk-band traffic light: these
# say which way a number moved a score, not how risky the customer is.
HELPS = "#2F6DA3"
HURTS = "#D97706"

BAND_TONE = {"Low": "#2F7A52", "Medium": "#D97706", "High": "#DC2626"}


def flatten(markup: str) -> str:
    """One line, no indentation, so Markdown leaves the HTML alone."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


# --- officer screen -------------------------------------------------------
# Styles are in theme.py, where the band colours live.


def show(markup: str) -> None:
    """Draw markup from this module. The one place that opts into raw HTML."""
    st.markdown(markup, unsafe_allow_html=True)


BACKDROP = Path(__file__).resolve().parent.parent / "assets" / "gradient-bg.jpg"


@lru_cache(maxsize=1)
def _backdrop_uri() -> str:
    return "data:image/jpeg;base64," + base64.b64encode(BACKDROP.read_bytes()).decode("ascii")


def stylesheet(name: str) -> None:
    """Inject ui/<name>.css for the view being drawn; `__BACKDROP__` is the gradient photo."""
    css = Path(__file__).with_name(name).read_text(encoding="utf-8")
    css = css.replace("__BACKDROP__", _backdrop_uri())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def band_badge(band: str | None) -> str:
    if band not in BAND_ICONS:
        return ""
    return (
        f'<span class="officer-badge officer-badge--{band}">'
        f"{BAND_ICONS[band]} {escape(band)} risk</span>"
    )


def card_head(
    title: str, meta: str, band: str | None = None, lift: bool = False
) -> str:
    """A card's heading: who it is about, a line of detail, and the model's band.

    `lift` marks a card that hover-raises (theme.py), for the ones with an action.
    """
    modifier = " officer-card-head--lift" if lift else ""
    return flatten(f"""
<div class="officer-card-head{modifier}">
<div>
<div class="officer-card-title">{escape(title)}</div>
<div class="officer-card-meta">{escape(meta)}</div>
</div>
{band_badge(band)}
</div>
""")


def quote(text: str) -> str:
    """The customer's own words, set apart from the model's."""
    return f'<blockquote class="officer-quote">{escape(text)}</blockquote>'


def reason_line(description: str, raises_risk: bool) -> str:
    arrow, side = ("▲", "up") if raises_risk else ("▼", "down")
    return f'<span class="officer-arrow--{side}">{arrow}</span> {html_safe(escape(description))}'
