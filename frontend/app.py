"""Credit Risk · an explainable, fair credit decision system.

Run the API first, then this, both from the repo root:

    cd backend && uvicorn app.main:app --port 8014
    streamlit run frontend/app.py

The frontend reads nothing from disk. Every screen is served by the API, which
decides what each role is sent rather than trusting the page not to draw it.

Choosing a role is not authentication. It opens a set of screens; it proves
nothing, and the API says the same. A real deployment would put an identity
provider in front of both.
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

ROLE_KEY = "role"
IDENTITY_KEY = "identity"

ICONS = {"officer": "🧑‍💼", "engineer": "🔬", "cardholder": "💳"}

# What each role is for, in the second person, since this is the sign-in choice.
INVITATIONS = {
    "officer": "Look up a customer, see whether their next payment is at risk, "
               "and record what you decided.",
    "engineer": "Audit the model before anything ships: accuracy, fairness, "
                "explanations, thresholds and monitoring.",
    "cardholder": "See whether your own next payment looks at risk, and what "
                  "would help most.",
}


def main() -> None:
    try:
        roles = {entry["role"]: entry for entry in api.roles()}
    except api.ServiceUnavailable as exception:
        st.title("Credit Risk")
        api.guard(exception)
        return

    chosen = st.session_state.get(ROLE_KEY)
    if chosen not in roles:
        _landing(roles)
        return

    _signed_in_bar(roles[chosen])

    if chosen == "officer":
        _officer()
    elif chosen == "engineer":
        _engineer()
    else:
        _cardholder()


# --- choosing a role ------------------------------------------------------


def _landing(roles: dict) -> None:
    """The way in. One card per role, because the three are peers.

    Rendered from the API's own role list, so a role cannot be offered here that
    the database would reject.
    """
    st.title("Credit Risk")
    st.subheader("Explainable, fair behavioural scoring")
    st.caption(
        "Scores existing cardholders for the risk of missing their next payment, "
        "explains every score in plain language, and audits itself for fairness."
    )
    st.divider()

    st.markdown("#### Sign in as")
    st.caption(
        "Choosing a role opens that person's screens. It is not authentication "
        "and proves nothing; a real deployment would sign you in properly."
    )

    for column, role in zip(st.columns(len(roles)), roles.values()):
        with column:
            _role_card(role)


def _role_card(role: dict) -> None:
    with st.container(border=True):
        st.markdown(f"## {ICONS.get(role['role'], '👤')}")
        st.markdown(f"### {role['label']}")
        st.write(INVITATIONS.get(role["role"], role["description"]))

        accounts = role["users"]
        st.caption(
            f"{accounts} account{'s' if accounts != 1 else ''} available"
            if accounts else "No accounts seeded yet"
        )

        if st.button(
            f"Continue as {role['label'].lower()}",
            key=f"pick_{role['role']}",
            width="stretch",
            type="primary",
            disabled=not accounts,
        ):
            st.session_state[ROLE_KEY] = role["role"]
            st.session_state.pop(IDENTITY_KEY, None)
            st.rerun()


def _signed_in_bar(role: dict) -> None:
    """Who you are, and the way back out, on every screen."""
    title, identity = st.columns([5, 2])
    title.title("Credit Risk")

    with identity:
        st.caption(f"{ICONS.get(role['role'], '👤')} Signed in as **{role['label']}**")
        if st.button("Switch role", width="stretch"):
            st.session_state.pop(ROLE_KEY, None)
            st.session_state.pop(IDENTITY_KEY, None)
            st.rerun()


def _who(role: str, label: str) -> str | None:
    """Which member of staff is acting, since a decision records who took it."""
    try:
        staff = api.users(role=role)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return None

    if not staff:
        st.warning(
            f"No {label.lower()} accounts exist. Seed them with: "
            "`cd backend && python -m app.db.seed`"
        )
        return None

    by_name = {
        (user.get("full_name") or user["display_name"]): user["email"]
        for user in staff
    }
    chosen = st.sidebar.selectbox(label, list(by_name))
    st.sidebar.caption(by_name[chosen])
    return by_name[chosen]


# --- the three screens ----------------------------------------------------


def _officer() -> None:
    email = _who("officer", "Officer")
    if email is None:
        return
    officer.render(email)


def _cardholder() -> None:
    st.subheader("Your account")
    cardholder.render()


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
