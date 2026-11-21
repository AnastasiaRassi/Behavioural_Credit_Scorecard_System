"""Right-hand utility drawer: navigation, sign out.

The toggle sits in the banner's `top_actions` container (app.py); the drawer is a
fixed panel below the banner. Its look is in drawer.css.
"""
import html
from collections.abc import Callable
from pathlib import Path

import streamlit as st

from ui import nav, profile

CSS = Path(__file__).with_name("drawer.css")
PANEL_KEY = "nav_panel"
TOGGLE_KEY = "nav_toggle"
SIGN_OUT_KEY = "nav_sign_out"
FOOTER_KEY = "nav_footer"

# Keep equal to the panel width in drawer.css. Wide screens shift the page right to
# make room; narrow ones let the panel overlay it.
BADGE_CAP = 99
WIDTH_PX = 320
_PUSH = f"""<style>
@media (min-width: 900px) {{
  :root {{ --nav-pad: {WIDTH_PX + 32}px; }}
  [data-testid="stMainBlockContainer"] {{ padding-left: {WIDTH_PX + 32}px !important; }}
}}
</style>"""


def render(
    display_name: str, role_label: str, on_sign_out: Callable[[], None],
    items: tuple[nav.Item, ...] = (),
    customer_label: str | None = None,
    on_close_customer: Callable[[], None] | None = None,
    messages_badge: int = 0,
    conversation_label: str | None = None,
    on_close_conversation: Callable[[], None] | None = None,
) -> None:
    """The toggle button, and the panel when open.

    `items` are the role's views above Profile. `customer_label` adds an indented
    entry under the first one: the customer an officer has opened.
    """
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    is_open = nav.drawer_is_open()
    st.button(
        ":material/close:" if is_open else ":material/menu:",
        key=TOGGLE_KEY, on_click=nav.toggle_drawer,
        help="Close menu" if is_open else "Open menu",
    )
    if is_open:
        st.markdown(_PUSH, unsafe_allow_html=True)
        _panel(display_name, role_label, on_sign_out, items, customer_label, on_close_customer, messages_badge,
               conversation_label, on_close_conversation)


def _panel(
    display_name: str, role_label: str, on_sign_out: Callable[[], None],
    items: tuple[nav.Item, ...],
    customer_label: str | None,
    on_close_customer: Callable[[], None] | None,
    messages_badge: int,
    conversation_label: str | None = None,
    on_close_conversation: Callable[[], None] | None = None,
) -> None:
    if messages_badge:
        # A button label can't hold markup, so the bubble is drawn by CSS.
        shown = "99+" if messages_badge > BADGE_CAP else messages_badge
        st.markdown(
            f'<style>.st-key-nav_{nav.CHAT} button::after{{content:"{shown}"}}</style>',
            unsafe_allow_html=True,
        )
    with st.container(key=PANEL_KEY):
        st.markdown(
            f'<div class="nav-user"><div class="nav-badge">'
            f"{html.escape(profile.initials(display_name))}</div><div>"
            f'<div class="nav-name">{html.escape(display_name)}</div>'
            f'<div class="nav-role">{html.escape(role_label)}</div></div></div>'
            '<div class="nav-label">Menu</div>',
            unsafe_allow_html=True,
        )
        active = nav.current()
        for position, item in enumerate((*items, nav.PROFILE_ITEM, nav.REPORT_ITEM)):
            st.button(
                f"{item.icon} {item.label}", key=f"nav_{item.view}", width="stretch",
                type="primary" if item.view == active else "secondary",
                on_click=nav.go, args=(item.view,),
            )
            if item.view == nav.CHAT and conversation_label:
                with st.container(key="nav_conversation_block"):
                    name, close = st.columns([5, 1], vertical_alignment="center")
                    name.button(
                        f":material/forum: {conversation_label}", key=f"nav_{nav.CONVERSATION}",
                        width="stretch",
                        type="primary" if active == nav.CONVERSATION else "secondary",
                        on_click=nav.go, args=(nav.CONVERSATION,),
                    )
                    close.button(
                        ":material/close:", key="nav_conversation_close",
                        help="Close this conversation", on_click=on_close_conversation,
                    )
            if position == 0 and customer_label:
                with st.container(key="nav_customer_block"):
                    name, close = st.columns([5, 1], vertical_alignment="center")
                    name.button(
                        f":material/person: {customer_label}", key=f"nav_{nav.CUSTOMER}",
                        width="stretch",
                        type="primary" if active == nav.CUSTOMER else "secondary",
                        on_click=nav.go, args=(nav.CUSTOMER,),
                    )
                    close.button(
                        ":material/close:", key="nav_customer_close", help="Close this user",
                        on_click=on_close_customer,
                    )
                    st.button(
                        f"{nav.PREDICTION_ITEM.icon} {nav.PREDICTION_ITEM.label}",
                        key=f"nav_{nav.PREDICTION}", width="stretch",
                        type="primary" if active == nav.PREDICTION else "secondary",
                        on_click=nav.go, args=(nav.PREDICTION,),
                    )
        with st.container(key=FOOTER_KEY):
            st.divider()
            st.button(
                ":material/logout: Sign Out", key=SIGN_OUT_KEY, width="stretch",
                on_click=_sign_out, args=(on_sign_out,),
            )


def _sign_out(on_sign_out: Callable[[], None]) -> None:
    nav.close_drawer()
    on_sign_out()
