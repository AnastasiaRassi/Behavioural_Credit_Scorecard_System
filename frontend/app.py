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
import json
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
from tabs import drift, explain, fairness, performance, registry, segmentation, threshold  # noqa: E402
from ui import drawer, nav, profile  # noqa: E402
from ui.components import stylesheet  # noqa: E402
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


def _browser_script(js: str) -> None:
    components.html(f"<script>{js}</script>", height=0)


def _store_token(token: str, reload: bool = False) -> None:
    """Keep the token in this tab's sessionStorage, so other tabs stay separate."""
    then = "p.location.reload();" if reload else ""
    _browser_script(
        "const p=window.parent;"
        f"p.sessionStorage.setItem('{COOKIE}',{json.dumps(token)});"
        f"p.document.cookie='{COOKIE}=; path=/; max-age=0';{then}"
    )


def _clear_token() -> None:
    _browser_script(
        "const p=window.parent;p.sessionStorage.removeItem('" + COOKIE + "');"
        f"p.document.cookie='{COOKIE}=; path=/; max-age=0';"
    )


def _hand_over_token() -> None:
    """Fresh page: pass this tab's token to the server via a 30s cookie, then reload."""
    _browser_script(
        "const p=window.parent;"
        f"const t=p.sessionStorage.getItem('{COOKIE}');"
        f"if(t&&!p.document.cookie.includes('{COOKIE}=')){{"
        f"p.document.cookie='{COOKIE}='+t+'; path=/; max-age=30; SameSite=Strict';"
        "p.location.reload();}"
    )


def _restore_session() -> tuple[dict | None, bool]:
    """Re-validate the handed-over token; the bool says whether one was offered."""
    token = st.context.cookies.get(COOKIE)
    if not token:
        return None, False
    try:
        return api.resume(token), True
    except api.ServiceUnavailable:
        return None, True


def _seed_location(session: dict) -> None:
    """After a reload, reopen the page the URL says we were on."""
    params = st.query_params
    if params.get("page") == "profile":
        st.session_state[profile.PROFILE_OPEN_KEY] = True
    elif session["role"] == "officer" and params.get("customer"):
        st.session_state[officer.CUSTOMER_KEY] = params["customer"]
        st.session_state[nav.VIEW_KEY] = nav.CUSTOMER
    elif params.get("view") == nav.CHAT:
        st.session_state[nav.VIEW_KEY] = nav.CHAT


def _sync_location() -> None:
    """Mirror the open page into the URL so a reload can restore it."""
    wanted = {}
    if profile.is_open():
        wanted["page"] = "profile"
    elif nav.current() == nav.CUSTOMER and st.session_state.get(officer.CUSTOMER_KEY):
        wanted["customer"] = str(st.session_state[officer.CUSTOMER_KEY])
    elif nav.current() == nav.CHAT:
        wanted["view"] = nav.CHAT
    if dict(st.query_params) != wanted:
        st.query_params.clear()
        st.query_params.update(wanted)


def main() -> None:
    session = st.session_state.get(SESSION_KEY)
    if session is None and st.session_state.pop(CLEAR_COOKIE_KEY, False):
        _clear_token()
    elif session is None and not st.session_state.get(RESTORE_TRIED_KEY):
        # Try once per page load: st.context.cookies is frozen at load, so a later
        # rerun would "restore" the account that just signed out.
        st.session_state[RESTORE_TRIED_KEY] = True
        session, offered = _restore_session()
        if session is not None:
            st.session_state[SESSION_KEY] = session
            _seed_location(session)
        elif offered:
            _clear_token()  # stale token: drop it so we don't reload forever
        else:
            _hand_over_token()

    if session is None:
        st.query_params.clear()
        _sign_in()
        return

    if st.session_state.pop(JUST_SIGNED_IN_KEY, False):
        # Reload once after signing in: Streamlit otherwise leaves the old form drawn
        # over the next screen. The fresh page restores the session from the cookie.
        _store_token(session["token"], reload=True)
        st.stop()

    _store_token(session["token"])
    _sync_location()
    stylesheet("backdrop.css")
    if session["role"] == "engineer" and not profile.is_open():
        stylesheet("workspace.css")

    # Drawn by the screen, not here: the officer and cardholder screens put it at
    # the top of their main column so Loana's card can start at the top of the page.
    def header() -> None:
        _signed_in_bar(session)

    if profile.is_open():
        header()
        nav.back_button("Back to Dashboard" if session["role"] == "cardholder" else "Back")
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
        # Cool slate bar; the logo's white edge pixels vanish via multiply blend.
        ".ct-banner{position:fixed;top:0;left:0;right:0;width:100vw;height:84px;"
        "z-index:1000000;display:flex;align-items:center;gap:1rem;"
        "padding:0 1.75rem 0 5.25rem;margin:0;border-radius:0;box-sizing:border-box;"
        "background:rgba(255,255,255,.45);backdrop-filter:blur(20px) saturate(160%);"
        "-webkit-backdrop-filter:blur(20px) saturate(160%);border-bottom:1px solid rgba(255,255,255,.75);"
        "border-top:3px solid #00A669;"
        "box-shadow:inset 0 -1px 1px rgba(255,255,255,.7),0 8px 24px rgba(15,23,42,.06)}"
        ".ct-banner img{height:56px;width:auto;display:block;background:transparent!important;"
        "border:none!important;box-shadow:none!important;padding:0!important;"
        "border-radius:0!important;mix-blend-mode:multiply}"
        ".ct-banner span{font-family:'Inter',sans-serif;font-weight:600;"
        "font-size:clamp(1rem,2vw,1.5rem);letter-spacing:.5px;color:#0F172A;"
        "text-transform:uppercase;line-height:1;white-space:nowrap}"
        # Page actions (officer back button) and the page crumb: the banner's right side.
        ".st-key-top_actions{position:fixed;top:0;right:13.5rem;height:84px;"
        "z-index:1000001;display:flex;flex-direction:row;align-items:center;"
        "gap:.5rem;width:auto!important}"
        ".ct-crumb{position:fixed;top:0;right:1.75rem;height:84px;z-index:1000001;display:flex;"
        "align-items:center;font-weight:700;font-size:1.05rem;color:#0F172A}"
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
        '[data-testid="stMainBlockContainer"]{padding-top:calc(84px + 1rem)!important}'
        # Style-only and fixed-position markdown would each add a flex gap above the page.
        '[data-testid="stElementContainer"]:has(>[data-testid="stMarkdown"] [data-testid="stMarkdownContainer"]>style:only-child){display:none}'
        '[data-testid="stElementContainer"]:has(.ct-banner),[data-testid="stElementContainer"]:has(.ct-crumb){position:absolute}'
        "</style>"
        f'<div class="ct-banner"><img src="data:image/png;base64,{icon}" alt="CrediTrait">'
        "<span>Explainable Credit Decisioning</span></div>"
    )


def _crumb(session: dict) -> str:
    """The page name shown at the banner's right."""
    if session["role"] == "officer":
        return nav.OFFICER_TITLES.get(nav.current(), "")
    if profile.is_open() or session["role"] == "cardholder":
        return nav.TITLES.get(nav.current(), "")
    return "Model monitoring"


def _signed_in_bar(session: dict) -> None:
    """The banner, then who you are and the way back out, on every screen."""
    st.markdown(_banner_html(), unsafe_allow_html=True)
    crumb = _crumb(session)
    if crumb:
        st.markdown(f'<div class="ct-crumb">{crumb}</div>', unsafe_allow_html=True)

    # Pinned into the banner's right side by the .st-key-top_actions rules in _banner_html.
    with st.container(key="top_actions"):
        drawer.render(
            session["display_name"], LABELS.get(session["role"], session["role"]), _sign_out,
            items=_nav_items(session),
            customer_label=officer.customer_label() if session["role"] == "officer" else None,
            on_close_customer=officer.close_customer,
            messages_badge=officer.waiting_count() if session["role"] == "officer" else 0,
            conversation_label=officer.conversation_label() if session["role"] == "officer" else None,
            on_close_conversation=officer.close_conversation,
        )


def _nav_items(session: dict) -> tuple[nav.Item, ...]:
    if session["role"] == "officer":
        return nav.OFFICER_ITEMS
    if session["role"] == "cardholder" and session["customer_id"] is not None:
        return nav.ACCOUNT_ITEMS
    return ()


def _sign_out() -> None:
    # The token is the whole session, so dropping it is the sign-out.
    # Cached reads go with it: they were fetched for this caller.
    st.session_state.pop(SESSION_KEY, None)
    st.session_state[CLEAR_COOKIE_KEY] = True
    for key in (profile.PROFILE_OPEN_KEY, nav.VIEW_KEY, nav.DRAWER_KEY):
        st.session_state.pop(key, None)
    st.cache_data.clear()


# --- the three screens ----------------------------------------------------


def _officer(session: dict, header: Callable[[], None]) -> None:
    # The acting officer is whoever signed in, not a sidebar choice: a decision
    # records who took it, and that must not be selectable.
    officer.render(session["email"], session["token"], header)


def _cardholder(session: dict, header: Callable[[], None]) -> None:
    if session["customer_id"] is None:
        header()
        st.subheader("Your account")
        st.info(
            "This account is not registered against a customer yet, so there is "
            "no scored account to show. Link it with: "
            "`cd backend && python -m app.db.seed`"
        )
        return
    cardholder.render(session["token"], session["customer_id"], header)


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
        "Threshold", "Segments", "Registry", "Monitoring",
    ]
    renderers = [
        performance.render, fairness.render, explain.render,
        threshold.render, segmentation.render, registry.render, drift.render,
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
