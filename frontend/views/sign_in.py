"""The sign-in screen: one frosted-glass card centred on the dashboard's gradient.

The card holds the logo, the form and nothing else. Its styling lives in
`ui/auth.css`, injected by `ui/auth_style.py`, and every rule in it is held behind
`body:has(.st-key-signin_card)`, so none of it reaches the dashboard once a
session exists.

Nothing here touches a credential: `render` is handed the function that does,
and shows whatever that function says went wrong.

Known limit: the message clears on the next attempt rather than on the first
keystroke. Enter has to submit the form, that needs `st.form`, and Streamlit
does not rerun on a widget change inside one, so there is no edit to react to.
"""
from collections.abc import Callable

import streamlit as st

from ui import auth_style

PRODUCT_NAME = "CrediTrait"

# What a failure says. Never a status code, a URL or a body: those go to the log
# in api.py, because nothing here is a problem the person reading it can fix.
EMPTY_EMAIL = "Enter your email."
EMPTY_PASSWORD = "Enter your password."
# One message whichever half was wrong: naming it would tell someone which
# accounts exist.
REJECTED = "That email and password don't match. Check both and try again."

# Set by app.py between runs. The pending credential survives exactly one rerun,
# which is what lets the button repaint as "Signing in…" before the attempt.
PENDING_KEY = "signin_pending"
ERROR_KEY = "signin_error"

_CHEVRON = """
<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"
     xmlns="http://www.w3.org/2000/svg">
<path d="M2.6 4.6 L6 8 L9.4 4.6" stroke="currentColor" stroke-width="1.6"
stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

_ALERT_ICON = """
<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"
     xmlns="http://www.w3.org/2000/svg">
<circle cx="7" cy="7" r="6.1" stroke="#8A3B2E" stroke-width="1.3"/>
<line x1="7" y1="3.7" x2="7" y2="7.8" stroke="#8A3B2E" stroke-width="1.4"
stroke-linecap="round"/>
<circle cx="7" cy="10.2" r="0.85" fill="#8A3B2E"/>
</svg>
"""


def render(attempt: Callable[[str, str], tuple[str, str] | None]) -> None:
    """Draw the screen.

    `attempt` owns the credentials and returns None on success, or the field at
    fault and the one line to show. Nothing here sees a token.
    """
    auth_style.inject()

    # Set on the run before this one, by a submit. Popped here
    # so a reload cannot replay it.
    pending = st.session_state.pop(PENDING_KEY, None)
    failure = st.session_state.get(ERROR_KEY)

    with st.container(key="hero"):
        st.markdown(_showcase(), unsafe_allow_html=True)

        # Widgets cannot live inside raw HTML, so the card is a keyed container
        # and the stylesheet reaches it through the .st-key-signin_card class.
        with st.container(key="signin_card"):
            st.markdown(_card_header(), unsafe_allow_html=True)
            with st.form("sign_in", border=False):
                with st.container(key="field_email"):
                    email = st.text_input("Email", placeholder="you@company.com")
                with st.container(key="field_password"):
                    password = st.text_input("Password", type="password")

                # Display only: a cookie always keeps you in for the token's 8h; no reset flow yet.
                with st.container(key="signin_row", horizontal=True):
                    st.checkbox("Remember me", key="signin_remember")
                    st.markdown(
                        '<a class="signin-forgot" href="#">Forgot password?</a>',
                        unsafe_allow_html=True,
                    )

                if failure:
                    st.markdown(_alert(*failure), unsafe_allow_html=True)

                submitted = st.form_submit_button(
                    "Signing in…" if pending else "Sign in",
                    type="primary",
                    disabled=bool(pending),
                )


    if submitted:
        # Hand the credential to the next run, so the button can repaint as
        # "Signing in…" before anything slow happens.
        st.session_state.pop(ERROR_KEY, None)
        st.session_state[PENDING_KEY] = (email, password)
        st.rerun()

    if pending:
        # A success ends in st.rerun() inside `attempt`, so nothing returns.
        outcome = attempt(*pending)
        if outcome is not None:
            st.session_state[ERROR_KEY] = outcome
            st.rerun()


def _card_header() -> str:
    return _flatten(f"""
<div class="signin-logo" role="img" aria-label="{PRODUCT_NAME}"></div>
<h2 class="signin-title">Welcome Back</h2>
<p class="signin-subtitle">Enter your credentials to access {PRODUCT_NAME}</p>
""")


def _icon(path: str) -> str:
    return (
        '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{path}</svg>'
    )


# (icon, title, line). Written for the person signing in, not the engineer.
_FEATURES = (
    (
        '<path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"/>'
        '<path d="M19 15l.8 1.9 1.9.8-1.9.8L19 20.5l-.8-1.9-1.9-.8 1.9-.8z"/>',
        "AI that explains itself",
        "Use Loana AI to understand the reasoning behind every score "
        "with clear explanations tied to real account behaviour.",
    ),
    (
        '<path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6z"/>'
        '<path d="M8.5 12.2l2.4 2.4 4.6-4.8"/>',
        "Careful, fair decisions",
        "Calibrated risk signals backed by clear reasoning and "
        "fairness checks across customer groups.",
    ),
    (
        '<path d="M3 20h18"/><path d="M5 16l4-5 4 3 6-8"/><path d="M15 6h4v4"/>',
        "Track loans and behaviour",
        "See payment trends and credit usage over time so early "
        "warning signs never go unnoticed.",
    ),
    (
        '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 9.5h8M8 12.5h5"/>',
        "Chat with AI or a person",
        "Get quick answers from Loana AI, or start a direct thread with "
        "a loan officer when a closer look is needed.",
    ),
)


def _showcase() -> str:
    """Four glass cards over the photograph; styled in ui/auth.css."""
    cards = "".join(
        f'<div class="feature-card">{_icon(icon)}<h3>{title}</h3><p>{line}</p></div>'
        for icon, title, line in _FEATURES
    )
    return f'<div class="signin-showcase">{cards}</div>'


def _alert(field: str, message: str) -> str:
    """The one line above the button. `field` marks which input to outline."""
    return _flatten(f"""
<div class="signin-error" data-field="{field}" role="alert">
{_ALERT_ICON}<span>{message}</span>
</div>
""")


def _flatten(markup: str) -> str:
    """One line, no indentation.

    Streamlit renders this through Markdown, which ends an HTML block at a blank
    line and then reads the four-space-indented line after it as a code fence.
    The SVG would print as source. Whitespace means nothing in the markup here.
    """
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())
