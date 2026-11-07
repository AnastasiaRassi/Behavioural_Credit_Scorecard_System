"""The sign-in screen: a 40/60 split, the logo on the left and the form on the right.

The left 40% is a solid panel holding the CrediTrait logo. The right 60% is a
faint office photograph with the sign-in card floating over it. The card holds
the form and nothing else; the story sections below the fold are unchanged.

Layout and colour live in `ui/login.css`, which is injected here with the photograph
and logo inlined as base64 so the page needs no static file server. Every rule in
it is held behind `body:has(.st-key-signin_card)`, so none of it reaches the
dashboard once a session exists.

Nothing here touches a credential: `render` is handed the function that does,
and shows whatever that function says went wrong.

Known limit: the message clears on the next attempt rather than on the first
keystroke. Enter has to submit the form, that needs `st.form`, and Streamlit
does not rerun on a widget change inside one, so there is no edit to react to.
"""
import base64
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import streamlit as st


HERE = Path(__file__).resolve().parent
CSS_PATHS = (HERE.parent / "ui" / "login.css", HERE.parent / "ui" / "landing.css")
PHOTO_PATH = HERE.parent / "assets" / "login-bg.jpg"
LOGO_PATH = HERE.parent / "assets" / "creditrait-logo.png"

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
    st.markdown(_stylesheet(), unsafe_allow_html=True)

    # Set on the run before this one, by a submit. Popped here
    # so a reload cannot replay it.
    pending = st.session_state.pop(PENDING_KEY, None)
    failure = st.session_state.get(ERROR_KEY)

    with st.container(key="hero"):
        st.markdown(_brand_panel(), unsafe_allow_html=True)

        # Widgets cannot live inside raw HTML, so the card is a keyed container
        # and the stylesheet reaches it through the .st-key-signin_card class.
        with st.container(key="signin_card"):
            with st.form("sign_in", border=False):
                with st.container(key="field_email"):
                    email = st.text_input("Email", placeholder="you@company.com")
                with st.container(key="field_password"):
                    password = st.text_input("Password", type="password")

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


def _alert(field: str, message: str) -> str:
    """The one line above the button. `field` marks which input to outline."""
    return _flatten(f"""
<div class="signin-error" data-field="{field}" role="alert">
{_ALERT_ICON}<span>{message}</span>
</div>
""")


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    """The stylesheet with the photograph and logo inlined. Read once per process."""
    for path in (*CSS_PATHS, PHOTO_PATH, LOGO_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. The sign-in screen ships with it; restore it "
                "from version control."
            )

    photo = base64.b64encode(PHOTO_PATH.read_bytes()).decode("ascii")
    logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    css = "\n".join(path.read_text(encoding="utf-8") for path in CSS_PATHS)
    # Declared once and referenced as var(...): inlining the data URI at each use
    # put a copy of the image in the markup per use, on every rerun.
    root = (
        ":root { "
        f'--login-photo: url("data:image/jpeg;base64,{photo}"); '
        f'--brand-logo: url("data:image/png;base64,{logo}"); }}'
    )
    return f"<style>{root}\n{css}</style>"


def _brand_panel() -> str:
    """The left 40%: the logo, centred. A background image, so it costs no markup."""
    return _flatten(f"""
<span id="top"></span>
<div class="brand-panel">
<div class="brand-logo" role="img" aria-label="{PRODUCT_NAME}"></div>
</div>
""")


def _flatten(markup: str) -> str:
    """One line, no indentation.

    Streamlit renders this through Markdown, which ends an HTML block at a blank
    line and then reads the four-space-indented line after it as a code fence.
    The SVG would print as source. Whitespace means nothing in the markup here.
    """
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())
