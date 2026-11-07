"""The page frame shared by the officer and cardholder screens.

Main content takes 70% of the width and Loana AI the other 30%, side by side on
one screen. Both roles go through `columns()`, `panel()` and `chat_area()`, so the
assistant looks the same to a loan officer and to a customer. The green card is
styled in theme.py, keyed on `PANEL_KEY` and `CHAT_KEY`.
"""
import base64
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import streamlit as st

AVATAR = Path(__file__).resolve().parent.parent / "assets" / "loana.png"

# Streamlit turns a container key into the class `st-key-<key>`; theme.py styles it.
PANEL_KEY = "loana_panel"
CHAT_KEY = "loana_chat"

# Only the fallback: theme.py makes the chat fill whatever the card has left.
CHAT_HEIGHT = 420


def no_header() -> None:
    """Default for views drawn without a page header (tests)."""


def columns():
    """The 70/30 split: (main, assistant).

    Views draw the page header at the top of the main column, not above both
    columns, so Loana's card starts at the top of the page and is on screen
    whole, input included, without scrolling.
    """
    return st.columns([7, 3], gap="large")


@lru_cache(maxsize=1)
def _avatar_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(AVATAR.read_bytes()).decode("ascii")


@contextmanager
def panel():
    """Loana's card: the header, then whatever chat is drawn inside."""
    with st.container(key=PANEL_KEY):
        st.markdown(
            f'<div class="loana-head"><img src="{_avatar_uri()}" alt="Loana AI">'
            '<div><div class="loana-name">LOANA AI</div>'
            '<div class="loana-sub">Financial Health Copilot</div></div></div>',
            unsafe_allow_html=True,
        )
        yield


def chat_area():
    """The scrolling message list. Draw presets and the input below it, not in it."""
    return st.container(height=CHAT_HEIGHT, key=CHAT_KEY)
