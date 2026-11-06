"""Atrium · an explainable, fair credit decision system.

Run the API first, then this, both from the repo root:

    cd backend && uvicorn app.main:app --port 8014
    streamlit run frontend/app.py

The frontend reads nothing from disk. Every screen is served by the API, which
decides what each role is sent rather than trusting the page not to draw it.

Signing in exchanges a password for a signed token, and the role comes back from
the account. Which screens open is therefore decided by the database, not by the
person at the keyboard, and this module never sees a role it was not told.
"""
import sys
from pathlib import Path

import streamlit as st

# Allow `import api` and `import views.*` when Streamlit runs this by path.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import api  # noqa: E402
import theme  # noqa: E402
from tabs import drift, explain, fairness, performance, registry, threshold  # noqa: E402
from views import cardholder, officer, sign_in  # noqa: E402

# The signed-in session: the token plus who the API says the caller is.
SESSION_KEY = "session"

ICONS = {"officer": "🧑‍💼", "engineer": "🔬", "cardholder": "💳"}

LABELS = {
    "officer": "Loan officer",
    "engineer": "ML engineer",
    "cardholder": "Cardholder",
}

def main() -> None:
    session = st.session_state.get(SESSION_KEY)
    if session is None:
        _sign_in()
        return

    _signed_in_bar(session)

    if session["role"] == "officer":
        _officer(session)
    elif session["role"] == "engineer":
        _engineer()
    else:
        _cardholder(session)


# --- signing in -----------------------------------------------------------


def _sign_in() -> None:
    """The way in: an email and a password. The account decides what opens."""
    # The screen is in views/sign_in.py; the credentials never leave this module.
    sign_in.render(_attempt)


def _attempt(email: str, password: str) -> tuple[str, str] | None:
    """Try the credentials, keeping a session only if the API accepted them.

    Returns rather than draws: the screen decides where the message goes and
    which field to outline. None means it worked, and the rerun below replaces
    the sign-in page with whatever the account opens.
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

    st.rerun()
    return None


def _signed_in_bar(session: dict) -> None:
    """Who you are, and the way back out, on every screen."""
    title, identity = st.columns([5, 2])
    title.title("Atrium")

    with identity:
        st.caption(
            f"{ICONS.get(session['role'], '👤')} **{session['display_name']}** · "
            f"{LABELS.get(session['role'], session['role'])}"
        )
        if st.button("Sign out", width="stretch"):
            # The token is the whole session, so dropping it is the sign-out.
            # Cached reads go with it: they were fetched for this caller.
            st.session_state.pop(SESSION_KEY, None)
            st.cache_data.clear()
            st.rerun()


# --- the three screens ----------------------------------------------------


def _officer(session: dict) -> None:
    # The acting officer is whoever signed in, not a sidebar choice: a decision
    # records who took it, and that must not be selectable.
    officer.render(session["email"])


def _cardholder(session: dict) -> None:
    st.subheader("Your account")
    if session["customer_id"] is None:
        st.info(
            "This account is not registered against a customer yet, so there is "
            "no scored account to show. Link it with: "
            "`cd backend && python -m app.db.seed`"
        )
        return
    cardholder.render(session["token"], session["customer_id"])


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
    page_title="Atrium · Explainable Credit Decisions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme.apply()
main()
