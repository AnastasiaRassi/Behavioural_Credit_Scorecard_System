"""Which main view is showing, and whether the drawer is open. State only, no drawing.

Views live in session state, so switching is a rerun with no scrolling, and the
conversation held under its own keys is untouched.
"""
from dataclasses import dataclass

import streamlit as st

from ui import profile

VIEW_KEY = "active_view"
DRAWER_KEY = "nav_drawer_open"

DASHBOARD = "dashboard"
CHAT = "chat"
PROFILE = "profile"
CUSTOMER = "customer"
CONVERSATION = "conversation"


@dataclass(frozen=True)
class Item:
    view: str
    label: str
    icon: str


# Account views belong to a cardholder with a scored account.
ACCOUNT_ITEMS = (
    Item(DASHBOARD, "Account Overview", ":material/dashboard:"),
    Item(CHAT, "Messages", ":material/chat:"),
)
# Officers: the dashboard is the user filter. A chosen customer gets its own entry
# (CUSTOMER) that the drawer indents under Filter Users.
OFFICER_ITEMS = (
    Item(DASHBOARD, "Filter Users", ":material/filter_list:"),
    Item(CHAT, "Messages", ":material/chat:"),
)
TITLES = {DASHBOARD: "Account Overview", CHAT: "Messages", PROFILE: "Profile"}
OFFICER_TITLES = {
    DASHBOARD: "Filter Users", CHAT: "Messages", PROFILE: "Profile", CUSTOMER: "Customer account",
    CONVERSATION: "Conversation",
}
PROFILE_ITEM = Item(PROFILE, "Profile", ":material/person:")


def current() -> str:
    """The view in the main workspace; the profile page wins while it is open."""
    if profile.is_open():
        return PROFILE
    return st.session_state.get(VIEW_KEY, DASHBOARD)


def go(view: str) -> None:
    """Callback: show `view`. The drawer stays open, as it is a sidebar."""
    if view == PROFILE:
        profile.set_open(True)
    else:
        profile.set_open(False)
        st.session_state[VIEW_KEY] = view


def back_button(label: str = "Back to Dashboard") -> None:
    """The big rectangle at the top left of a sub-page; styled in backdrop.css."""
    st.button(
        f":material/arrow_back: {label}", key="back_to_dashboard",
        on_click=go, args=(DASHBOARD,),
    )


def drawer_is_open() -> bool:
    return bool(st.session_state.get(DRAWER_KEY))


def toggle_drawer() -> None:
    st.session_state[DRAWER_KEY] = not drawer_is_open()


def close_drawer() -> None:
    st.session_state[DRAWER_KEY] = False
