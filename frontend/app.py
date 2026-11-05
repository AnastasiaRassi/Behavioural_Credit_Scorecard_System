"""Credit Risk · an explainable, fair credit decision system.

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
from views import cardholder, officer  # noqa: E402

# The signed-in session: the token plus who the API says the caller is.
SESSION_KEY = "session"

ICONS = {"officer": "🧑‍💼", "engineer": "🔬", "cardholder": "💳"}

LABELS = {
    "officer": "Loan officer",
    "engineer": "ML engineer",
    "cardholder": "Cardholder",
}

# What each kind of account opens, so the sign-in page still explains the system
# to someone seeing it for the first time. Descriptive, not a choice to make.
OPENS = {
    "officer": "A worklist of scored cardholders, the reasons behind each score, "
               "and a decision to record.",
    "engineer": "The model audit: accuracy, fairness, explanations, thresholds "
                "and monitoring.",
    "cardholder": "One account — your own — and whether its next payment looks "
                  "at risk.",
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
    st.title("Credit Risk")
    st.caption("Explainable, fair behavioural scoring")

    middle = st.columns([1, 2, 1])[1]
    with middle:
        with st.container(border=True):
            st.markdown("#### Sign in")
            st.caption("Your account decides which screens open.")

            with st.form("sign_in", border=False):
                email = st.text_input("Email", placeholder="name@example.test")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Sign in", type="primary", width="stretch"
                )

            if submitted:
                _attempt(email, password)

        _what_accounts_open()


def _attempt(email: str, password: str) -> None:
    """Try the credentials, keeping a session only if the API accepted them."""
    if not email.strip() or not password:
        st.error("Enter both an email and a password.")
        return

    try:
        st.session_state[SESSION_KEY] = api.sign_in(email.strip(), password)
    except api.SignInRejected as exception:
        # The API's single message, deliberately: saying which half was wrong
        # would tell someone which accounts exist.
        st.error(str(exception))
        return
    except api.ServiceUnavailable as exception:
        st.error(str(exception))
        return

    st.rerun()


def _what_accounts_open() -> None:
    """What each kind of account sees, read from the API's own role list."""
    try:
        roles = api.roles()
    except api.ServiceUnavailable:
        # Signing in is the point of this page, so a service that cannot describe
        # its roles must not stop someone trying.
        return

    st.caption("The system has three kinds of account:")
    for role in roles:
        name = LABELS.get(role["role"], role["role"])
        st.markdown(
            f"{ICONS.get(role['role'], '👤')} **{name}** — "
            f"{OPENS.get(role['role'], role['description'])}"
        )


def _signed_in_bar(session: dict) -> None:
    """Who you are, and the way back out, on every screen."""
    title, identity = st.columns([5, 2])
    title.title("Credit Risk")

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
    page_title="Credit Risk · Explainable Scoring",
    page_icon="📊",
    layout="wide",
)

theme.apply()
main()
