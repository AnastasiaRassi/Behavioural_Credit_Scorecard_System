"""CrediTrait · an explainable, fair credit decision system.

Run the API first, then this, both from the repo root:

    cd backend && uvicorn app.main:app --port 8014
    streamlit run frontend/app.py

The frontend reads nothing from disk. Every screen is served by the API, which
decides what each role is sent rather than trusting the page not to draw it.

Signing in exchanges a password for a signed token, and the role comes back from
the account. Which screens open is therefore decided by the database, not by the
person at the keyboard, and this module never sees a role it was not told.
"""
import base64
import sys
from collections.abc import Callable
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Allow `import api` and `import views.*` when Streamlit runs this by path.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import api  # noqa: E402
import theme  # noqa: E402
from tabs import drift, explain, fairness, performance, registry, threshold  # noqa: E402
from ui import profile  # noqa: E402
from views import cardholder, officer, sign_in  # noqa: E402

# The signed-in session: the token plus who the API says the caller is.
SESSION_KEY = "session"
ICON_PATH = HERE / "assets" / "credit-logo-icon-transparent.png"

ICONS = {"officer": "🧑‍💼", "engineer": "🔬", "cardholder": "💳"}

LABELS = {
    "officer": "Loan officer",
    "engineer": "ML engineer",
    "cardholder": "Cardholder",
}

COOKIE = "ct_session"
CLEAR_COOKIE_KEY = "clear_session_cookie"
RESTORE_TRIED_KEY = "session_restore_tried"
JUST_SIGNED_IN_KEY = "just_signed_in"


def _cookie_script(
    token: str | None, max_age: int = 8 * 60 * 60, reload: bool = False
) -> None:
    """Set or clear the token cookie in the browser (Streamlit can't write cookies)."""
    value = f"{token}; path=/; max-age={max_age}; SameSite=Strict" if token else (
        "; path=/; max-age=0"
    )
    then = "window.parent.location.reload();" if reload else ""
    components.html(
        f"<script>window.parent.document.cookie='{COOKIE}='+'{value}';{then}</script>",
        height=0,
    )


def _restore_session() -> dict | None:
    """Reload keeps the user in: re-validate the token cookie against the API."""
    token = st.context.cookies.get(COOKIE)
    if not token:
        return None
    try:
        return api.resume(token)
    except api.ServiceUnavailable:
        return None


def _seed_location(session: dict) -> None:
    """After a reload, reopen the page the URL says we were on."""
    params = st.query_params
    if params.get("page") == "profile":
        st.session_state[profile.PROFILE_OPEN_KEY] = True
    elif session["role"] == "officer" and params.get("customer"):
        st.session_state[officer.CUSTOMER_KEY] = params["customer"]


def _sync_location() -> None:
    """Mirror the open page into the URL so a reload can restore it."""
    wanted = {}
    if profile.is_open():
        wanted["page"] = "profile"
    elif st.session_state.get(officer.CUSTOMER_KEY):
        wanted["customer"] = str(st.session_state[officer.CUSTOMER_KEY])
    if dict(st.query_params) != wanted:
        st.query_params.clear()
        st.query_params.update(wanted)


def main() -> None:
    session = st.session_state.get(SESSION_KEY)
    if session is None and st.session_state.pop(CLEAR_COOKIE_KEY, False):
        _cookie_script(None)
    elif session is None and not st.session_state.get(RESTORE_TRIED_KEY):
        # st.context.cookies is frozen at page load, so a later rerun would "restore"
        # the account that just signed out. Try once per page load only.
        st.session_state[RESTORE_TRIED_KEY] = True
        session = _restore_session()
        if session is not None:
            st.session_state[SESSION_KEY] = session
            _seed_location(session)

    if session is None:
        st.query_params.clear()
        _sign_in()
        return

    if st.session_state.pop(JUST_SIGNED_IN_KEY, False):
        # Reload once after signing in: Streamlit otherwise leaves the old form drawn
        # over the next screen. The fresh page restores the session from the cookie.
        _cookie_script(session["token"], reload=True)
        st.stop()

    _cookie_script(session["token"])
    _sync_location()

    # Drawn by the screen, not here: the officer and cardholder screens put it at
    # the top of their main column so Loana's card can start at the top of the page.
    def header() -> None:
        _signed_in_bar(session)

    if profile.is_open():
        header()
        profile.render_page(session, LABELS.get(session["role"], session["role"]))
    elif session["role"] == "officer":
        _officer(session, header)
    elif session["role"] == "engineer":
        header()
        _engineer()
    else:
        _cardholder(session, header)


# --- signing in -----------------------------------------------------------


def _sign_in() -> None:
    """The way in: an email and a password. The account decides what opens."""
    # The screen is in views/sign_in.py; the credentials never leave this module.
    sign_in.render(_attempt)


def _attempt(email: str, password: str) -> tuple[str, str] | None:
    """Try the credentials, keeping a session only if the API accepted them.

    Returns rather than draws: the screen decides where the message goes and
    which field to outline. None means it worked, and the page reloads into
    whatever the account opens.
    """
    if not email.strip():
        return "email", sign_in.EMPTY_EMAIL
    if not password:
        return "password", sign_in.EMPTY_PASSWORD

    try:
        st.session_state[SESSION_KEY] = api.sign_in(email.strip(), password)
    except api.SignInRejected:
        return "both", sign_in.REJECTED
    except api.ServiceUnavailable as exception:
        # str() is the one sentence meant for a person; the status code and the
        # body are already in the log (see api.ServiceUnavailable).
        return "", str(exception)

    st.session_state[JUST_SIGNED_IN_KEY] = True
    st.rerun()
    return None


@st.cache_resource
def _banner_html() -> str:
    """Fixed full-width navbar. Read once per process."""
    icon = base64.b64encode(ICON_PATH.read_bytes()).decode("ascii")
    return (
        "<style>"
        # Fixed to the viewport, so column and container padding can't box it in.
        # The pale bar matches the logo's white edge pixels so they vanish.
        ".ct-banner{position:fixed;top:0;left:0;right:0;width:100vw;height:84px;"
        "z-index:1000000;display:flex;align-items:center;gap:1rem;"
        "padding:0 1.75rem;margin:0;border-radius:0;box-sizing:border-box;"
        "background:#FFFFFF;border-bottom:1px solid #E2E8F0;"
        "border-top:3px solid #059669;box-shadow:0 1px 3px rgba(15,23,42,.06)}"
        ".ct-banner img{height:56px;width:auto;display:block;background:transparent!important;"
        "border:none!important;box-shadow:none!important;padding:0!important;"
        "border-radius:0!important;mix-blend-mode:multiply}"
        ".ct-banner span{font-family:'Inter',sans-serif;font-weight:600;"
        "font-size:clamp(1rem,2vw,1.5rem);letter-spacing:.5px;color:#1C1917;"
        "text-transform:uppercase;line-height:1;white-space:nowrap}"
        # Badge and actions: one row on the banner's right, ghost-style buttons.
        ".st-key-top_actions{position:fixed;top:0;right:1.75rem;height:84px;"
        "z-index:1000001;display:flex;flex-direction:row;align-items:center;"
        "gap:.5rem;width:auto!important}"
        ".st-key-top_actions>*{width:auto!important;flex:none}"
        ".st-key-top_actions p{margin:0;font-size:13px;color:rgba(28,25,23,.65)}"
        # Ghost style for the back button only; the profile menu styles itself.
        ".st-key-officer_close button{background:transparent;border:1px solid rgba(28,25,23,.25);"
        "border-radius:8px;padding:4px 10px;min-height:0;line-height:1.4;"
        "transition:background .15s,border-color .15s}"
        ".st-key-officer_close button p{font-size:13px;font-weight:500;color:#1C1917}"
        ".st-key-officer_close button:hover,.st-key-officer_close button:focus-visible{"
        "background:rgba(28,25,23,.08);border-color:rgba(28,25,23,.5)}"
        # Content starts below the bar.
        '[data-testid="stMainBlockContainer"]{padding-top:7rem!important}'
        "</style>"
        f'<div class="ct-banner"><img src="data:image/png;base64,{icon}" alt="CrediTrait">'
        "<span>Explainable Credit Decisioning</span></div>"
    )


def _signed_in_bar(session: dict) -> None:
    """The banner, then who you are and the way back out, on every screen."""
    st.markdown(_banner_html(), unsafe_allow_html=True)

    # Pinned into the banner's right side by the .st-key-top_actions rules in _banner_html.
    with st.container(key="top_actions"):
        if session["role"] == "officer" and not profile.is_open():
            officer.back_button()
        profile.render(
            session["display_name"], LABELS.get(session["role"], session["role"]), _sign_out
        )


def _sign_out() -> None:
    # The token is the whole session, so dropping it is the sign-out.
    # Cached reads go with it: they were fetched for this caller.
    st.session_state.pop(SESSION_KEY, None)
    st.session_state[CLEAR_COOKIE_KEY] = True
    st.session_state.pop(profile.PROFILE_OPEN_KEY, None)
    st.cache_data.clear()


# --- the three screens ----------------------------------------------------


def _officer(session: dict, header: Callable[[], None]) -> None:
    # The acting officer is whoever signed in, not a sidebar choice: a decision
    # records who took it, and that must not be selectable.
    officer.render(session["email"], session["token"], header)


def _cardholder(session: dict, header: Callable[[], None]) -> None:
    def titled() -> None:
        header()
        st.subheader("Your account")

    if session["customer_id"] is None:
        titled()
        st.info(
            "This account is not registered against a customer yet, so there is "
            "no scored account to show. Link it with: "
            "`cd backend && python -m app.db.seed`"
        )
        return
    cardholder.render(session["token"], session["customer_id"], titled)


def _engineer() -> None:
    try:
        settings = api.config()
        colour, label, reason = api.health()
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    left, right = st.columns([3, 1])
    left.markdown(
        f"**{settings['model_version']}** &nbsp;·&nbsp; "
        f"trained {settings['trained']} &nbsp;·&nbsp; "
        f"{colour} **{label}**"
    )
    right.caption(reason)

    st.caption(
        f"{settings['training_rows']:,} training rows &nbsp;·&nbsp; "
        f"{settings['n_features']} features &nbsp;·&nbsp; "
        f"seeds {'/'.join(str(seed) for seed in settings['seeds'])} &nbsp;·&nbsp; "
        f"evaluated on {settings['test_rows']:,} held-out customers"
    )
    st.divider()

    names = [
        "Performance", "Fairness", "Explainability",
        "Threshold", "Registry", "Monitoring",
    ]
    renderers = [
        performance.render, fairness.render, explain.render,
        threshold.render, registry.render, drift.render,
    ]
    for tab, render in zip(st.tabs(names), renderers):
        with tab:
            render()


st.set_page_config(
    page_title="CrediTrait · Explainable Credit Decisions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.apply()
main()
