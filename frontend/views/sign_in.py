"""The sign-in screen: one photograph, one card.

The people who meet this screen are loan officers, product people and customers,
so it is the one place in the product that reads as a brochure rather than a
console. A view straight up between four glass towers leaves a plus-shaped gap
of sky; the card sits where the two arms of that gap cross, and two quiet panels
sit in the horizontal arm either side of it.

Layout and colour live in `ui/login.css`, which is injected here with the
photograph inlined as base64 so the page needs no static file server. Every rule
in it is held behind `body:has(.st-key-signin_card)`, so none of it reaches the
dashboard once a session exists.

Nothing here touches a credential: `render` is handed the function that does,
and shows whatever that function says went wrong.

Known limit: the message clears on the next attempt rather than on the first
keystroke. Enter has to submit the form, that needs `st.form`, and Streamlit
does not rerun on a widget change inside one, so there is no edit to react to.
"""
import base64
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ui.components import flatten
from views import landing

HERE = Path(__file__).resolve().parent
CSS_PATHS = (HERE.parent / "ui" / "login.css", HERE.parent / "ui" / "landing.css")
PHOTO_PATH = HERE.parent / "assets" / "login-bg.jpg"

# What both stylesheets write where the inlined photograph goes.

PRODUCT_NAME = "CrediTrait"
DESCRIPTION = (
    "Track loans and applications, and see the reasons behind every AI decision."
)
HEADING = "Sign in"
HELPER = "No account yet? Your administrator can set one up."

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

# DEMO_MODE=1 puts a one-click row under the form. The password matches the one
# `python -m app.db.seed` sets for these accounts in the same mode.
DEMO_PASSWORD = "demo"

# Only the two roles someone arriving at this page would expect to be. The
# engineer screens are an internal tool and listing them here read as if the
# public were being offered the model's own controls.
DEMO_ACCOUNTS = (
    ("Loan officer", "officer@demo.test"),
    ("Cardholder", "cardholder@demo.test"),
)

# Added to the row by the footer link, which sets ?internal=1. Convenience
# only, never a boundary: what a session may open is decided by the role the
# API returns for the account, not by how the page was reached.
INTERNAL_ACCOUNTS = (("ML engineer", "engineer@demo.test"),)
INTERNAL_PARAM = "internal"

# Four rounded squares around a plus-shaped gap: the view up between the towers.
_LOGO = """
<svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true"
     xmlns="http://www.w3.org/2000/svg">
<rect x="0.8" y="0.8" width="10.4" height="10.4" rx="2.6"
fill="rgba(74,107,133,0.10)" stroke="#4A6B85" stroke-width="1.5"/>
<rect x="16.8" y="0.8" width="10.4" height="10.4" rx="2.6"
fill="rgba(74,107,133,0.10)" stroke="#4A6B85" stroke-width="1.5"/>
<rect x="0.8" y="16.8" width="10.4" height="10.4" rx="2.6"
fill="rgba(74,107,133,0.10)" stroke="#4A6B85" stroke-width="1.5"/>
<rect x="16.8" y="16.8" width="10.4" height="10.4" rx="2.6"
fill="rgba(74,107,133,0.10)" stroke="#4A6B85" stroke-width="1.5"/>
<circle cx="14" cy="14" r="2.1" fill="#D9A462"/>
</svg>
"""

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

# How today's queue splits by risk band. Real proportions, from the live
# queue's own summary: 2,224 Low, 1,338 Medium, 938 High of 4,500 accounts.
# The earlier panel drew an Applied → Repaying pipeline, which this system does
# not have: it scores existing card accounts from their payment history, so
# there is no application to track and the panel was claiming a feature.
_TRACK = """
<svg viewBox="0 0 278 50" fill="none" aria-hidden="true"
     xmlns="http://www.w3.org/2000/svg">
<rect x="12" y="8" width="125" height="12" rx="3" fill="#2F7A52"/>
<rect x="137" y="8" width="75" height="12" rx="3" fill="#B8742A"/>
<rect x="212" y="8" width="54" height="12" rx="3" fill="#A6442F"/>
<text x="12" y="39" text-anchor="start">Low 2,224</text>
<text x="137" y="39" text-anchor="start">Medium 1,338</text>
<text x="266" y="39" text-anchor="end">High 938</text>
</svg>
"""

# What moved a score: right of the line helped, left of it hurt. An illustration
# of the shape of an explanation, so it carries no numbers.
_REASONS = """
<svg viewBox="0 0 278 78" fill="none" aria-hidden="true"
     xmlns="http://www.w3.org/2000/svg">
<line x1="139" y1="4" x2="139" y2="74" stroke="#C6D1DB" stroke-width="1"/>
<rect class="panel-bar" x="139" y="10" width="68" height="9" rx="4.5" fill="#2F6DA3"/>
<text x="131" y="18.5" text-anchor="end">Paid on time</text>
<rect class="panel-bar" x="87" y="34" width="52" height="9" rx="4.5" fill="#B8742A"/>
<text x="147" y="42.5" text-anchor="start">High balance</text>
<rect class="panel-bar" x="139" y="58" width="44" height="9" rx="4.5" fill="#2F6DA3"/>
<text x="131" y="66.5" text-anchor="end">Long history</text>
</svg>
"""


@lru_cache(maxsize=1)
def demo_mode() -> bool:
    """Whether to offer the one-click accounts.

    Read from .env as well as the environment, so DEMO_MODE=1 in the one file
    turns it on for the seed and for this page together, rather than needing
    the variable exported into whatever shell started Streamlit.
    """
    load_dotenv(HERE.parents[1] / ".env")
    return os.environ.get("DEMO_MODE") == "1"


def render(attempt: Callable[[str, str], tuple[str, str] | None]) -> None:
    """Draw the screen.

    `attempt` owns the credentials and returns None on success, or the field at
    fault and the one line to show. Nothing here sees a token.
    """
    st.markdown(_stylesheet(), unsafe_allow_html=True)

    # Set on the run before this one, by a submit or a demo button. Popped here
    # so a reload cannot replay it.
    pending = st.session_state.pop(PENDING_KEY, None)
    failure = st.session_state.get(ERROR_KEY)

    with st.container(key="hero"):
        st.markdown(_side_panels(), unsafe_allow_html=True)

        # Widgets cannot live inside raw HTML, so the card is a keyed container
        # and the stylesheet reaches it through the .st-key-signin_card class.
        with st.container(key="signin_card"):
            st.markdown(_masthead(), unsafe_allow_html=True)

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

            st.markdown(
                f'<p class="signin-helper">{HELPER}</p>', unsafe_allow_html=True
            )
            if demo_mode():
                _demo_row()

        st.markdown(_see_how(), unsafe_allow_html=True)

    st.markdown(landing.sections(), unsafe_allow_html=True)
    _closing_band()

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


def _see_how() -> str:
    """The way down to the sections: a quiet text button, not a blue link."""
    return _flatten(f"""
<div class="see-how-row">
<a class="see-how" href="#how-it-works">See how it works{_CHEVRON}</a>
</div>
""")


def _internal() -> bool:
    """Whether the footer link asked for the internal role."""
    return st.query_params.get(INTERNAL_PARAM) == "1"


def _demo_row(key: str = "demo_row") -> None:
    """One click per role, demo mode only. Hidden entirely when it is off."""
    accounts = DEMO_ACCOUNTS + (INTERNAL_ACCOUNTS if _internal() else ())
    st.markdown(
        '<p class="demo-label">Try a demo account</p>', unsafe_allow_html=True
    )
    with st.container(key=key):
        columns = st.columns(len(accounts))
        for column, (label, email) in zip(columns, accounts):
            if column.button(label, key=f"{key}_{email}", width="stretch"):
                st.session_state.pop(ERROR_KEY, None)
                st.session_state[PENDING_KEY] = (email, DEMO_PASSWORD)
                st.rerun()


def _footer() -> str:
    """The one quiet way to the internal view, under the closing card."""
    inside = _internal()
    href, words = ("?", "Leave internal access") if inside else ("?internal=1", "Internal access")
    return _flatten(f"""
<p class="page-footer"><a class="footer-link" href="{href}">{words}</a></p>
""")


def _closing_band() -> None:
    """The last band: one card back to the form, and the footer under it."""
    with st.container(key="closing"):
        st.markdown(landing.closing_band(), unsafe_allow_html=True)
        st.markdown(_footer(), unsafe_allow_html=True)


def _alert(field: str, message: str) -> str:
    """The one line above the button. `field` marks which input to outline."""
    return _flatten(f"""
<div class="signin-error" data-field="{field}" role="alert">
{_ALERT_ICON}<span>{message}</span>
</div>
""")


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    """The stylesheet with the photograph inlined. Read once per process."""
    for path in (*CSS_PATHS, PHOTO_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. The sign-in screen ships with it; restore it "
                "from version control."
            )

    photo = base64.b64encode(PHOTO_PATH.read_bytes()).decode("ascii")
    css = "\n".join(path.read_text(encoding="utf-8") for path in CSS_PATHS)
    # Declared once and referenced as var(--login-photo): inlining the data URI
    # at each use put a copy of the photograph in the markup per use, on every
    # rerun, and the stylesheet went past a megabyte.
    root = f':root {{ --login-photo: url("data:image/jpeg;base64,{photo}"); }}'
    return f"<style>{root}\n{css}</style>"


def _masthead() -> str:
    """Logo, product name, what the product is, and the heading over the form."""
    return _flatten(f"""
<div class="signin-brand">{_LOGO}<span class="signin-name">{PRODUCT_NAME}</span></div>
<p class="signin-desc">{DESCRIPTION}</p>
<h1 class="signin-heading">{HEADING}</h1>
""")


def _side_panels() -> str:
    """The two decorative panels in the horizontal arm of the sky gap."""
    return _flatten(f"""
<span id="top"></span>
<div class="sky-panel sky-panel--left" aria-hidden="true">
<p class="sky-panel-title">Today's queue</p>{_TRACK}
</div>
<div class="sky-panel sky-panel--right" aria-hidden="true">
<p class="sky-panel-title">Why this decision</p>{_REASONS}
</div>
""")


def _flatten(markup: str) -> str:
    """One line, no indentation.

    Streamlit renders this through Markdown, which ends an HTML block at a blank
    line and then reads the four-space-indented line after it as a code fence.
    The SVG would print as source. Whitespace means nothing in the markup here.
    """
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())
