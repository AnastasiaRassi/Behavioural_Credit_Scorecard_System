"""Which mode the officer's customer page is in, and the header that switches it.

One customer page with two modes rather than two drawer entries, so the risk
analysis always sits under the customer it belongs to. The header is a tab bar
and `panel()` is the surface the active tab opens onto; draw them one after the
other. Styles are in ui/customer_header.css.
"""
from html import escape

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from ui import nav
from ui.components import flatten, show, stylesheet

MODE_KEY = "officer_customer_mode"

RISK = "risk"
OVERVIEW = "overview"
# Button order, left to right.
MODES = (RISK, OVERVIEW)
LABELS = {RISK: "Risk assessment", OVERVIEW: "Overview"}
SUBTITLES = {
    RISK: "Chance of missing the next payment, the risk tier, and the behavioural "
          "drivers (SHAP) behind it.",
    OVERVIEW: "Six months of bills against payments, and the account summary.",
}
# The record is what an officer opened a customer to see before the model ran.
DEFAULT = OVERVIEW


def current() -> str:
    mode = st.session_state.get(MODE_KEY, DEFAULT)
    return mode if mode in MODES else DEFAULT


def go(mode: str) -> None:
    """Callback: show the customer page in `mode`."""
    st.session_state[MODE_KEY] = mode
    nav.go(nav.CUSTOMER)


def header(name: str, account_number: str, segment: str | None = None) -> None:
    """Customer name and account on the left, the mode tabs on the bottom-right edge."""
    stylesheet("customer_header.css")
    mode = current()
    meta = f"Account {escape(account_number)}"
    if segment:
        meta += f" · {escape(segment)} segment"
    with st.container(key="customer_header"):
        who, switch = st.columns([3, 2], vertical_alignment="bottom")
        with who:
            show(flatten(f"""
<div class="customer-header-name">{escape(name)}</div>
<div class="customer-header-meta">{meta}</div>
<div class="customer-header-subtitle"><b>{LABELS[mode]}</b> · {escape(SUBTITLES[mode])}</div>
"""))
        with switch, st.container(
            key="customer_mode", horizontal=True, horizontal_alignment="right", gap=None,
        ):
            for option in MODES:
                st.button(
                    LABELS[option], key=f"customer_mode_{option}",
                    type="primary" if option == mode else "secondary",
                    on_click=go, args=(option,),
                )


def panel() -> DeltaGenerator:
    """The white surface under the header that the active tab connects to."""
    return st.container(key="customer_view")
