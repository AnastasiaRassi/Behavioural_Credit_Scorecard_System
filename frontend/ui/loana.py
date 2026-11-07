"""Loana AI as a floating widget shared by the officer and cardholder screens.

Closed, it is a round avatar bubble in the bottom-right corner and the page uses the
full width. Open, it is a fixed chat card over the same corner. Both roles go through
`dock()`, so the assistant looks the same to a loan officer and to a customer. The
look is in theme.py, keyed on `PANEL_KEY`, `LAUNCH_KEY`, `CLOSE_KEY` and `CHAT_KEY`.
"""
import base64
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import streamlit as st

AVATAR = Path(__file__).resolve().parent.parent / "assets" / "loana.png"

# Streamlit turns a container key into the class `st-key-<key>`; theme.py styles it.
PANEL_KEY = "loana_panel"
CHAT_KEY = "loana_chat"
LAUNCH_KEY = "loana_launch"
CLOSE_KEY = "loana_close"
OPEN_STATE = "loana_open"

# Only the fallback: theme.py makes the chat fill whatever the card has left.
CHAT_HEIGHT = 520


def no_header() -> None:
    """Default for views drawn without a page header (tests)."""


def is_open() -> bool:
    return bool(st.session_state.get(OPEN_STATE, False))


def _set_open(value: bool) -> None:
    st.session_state[OPEN_STATE] = value


@lru_cache(maxsize=1)
def _avatar_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(AVATAR.read_bytes()).decode("ascii")


def dock(draw: Callable[[], None]) -> None:
    """Draw the bubble, or the chat card with `draw()` inside it when it is open."""
    if is_open():
        with panel():
            draw()
    else:
        _bubble()


def _bubble() -> None:
    with st.container(key=LAUNCH_KEY):
        # The image is a CSS background so the button stays a real, clickable button.
        st.markdown(
            f'<style>.st-key-{LAUNCH_KEY} button{{background-image:url("{_avatar_uri()}")}}</style>',
            unsafe_allow_html=True,
        )
        st.button(
            "Chat with Loana AI",
            key=f"{LAUNCH_KEY}_button",
            help="Ask Loana AI about this account",
            on_click=_set_open,
            args=(True,),
        )


@contextmanager
def panel():
    """Loana's card: the close button and header, then whatever chat is drawn inside."""
    with st.container(key=PANEL_KEY):
        st.button(
            "✕",
            key=CLOSE_KEY,
            help="Close chat",
            on_click=_set_open,
            args=(False,),
        )
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
