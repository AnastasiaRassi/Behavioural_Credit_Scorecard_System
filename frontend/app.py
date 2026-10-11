"""Credit Risk · an explainable, fair credit decision system.

Run the API first, then this, both from the repo root:

    cd backend && uvicorn app.main:app --port 8012
    streamlit run frontend/app.py

The frontend reads nothing from disk. Every screen is served by the API, which
decides what each role is sent rather than trusting the page not to draw it.

Picking a role is not authentication. It chooses which screens open; it does not
prove anything, and the API says the same. A real deployment would put an
identity provider in front of both.
"""
import sys
from pathlib import Path

import streamlit as st

# Allow `import api` and `import views.*` when Streamlit runs this by path.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import api  # noqa: E402
from tabs import drift, explain, fairness, performance, registry, threshold  # noqa: E402
from views import cardholder, officer  # noqa: E402

st.set_page_config(
    page_title="Credit Risk · Explainable Scoring",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    try:
        roles = api.roles()
    except api.ServiceUnavailable as exception:
        st.title("Credit Risk")
        api.guard(exception)
        return

    role = _role_picker(roles)

    if role == "officer":
        _officer()
    elif role == "engineer":
        _engineer()
    else:
        _cardholder()


def _role_picker(roles: list[dict]) -> str:
    """Rendered from the API's own vocabulary, so a role cannot appear here that
    the schema would reject."""
    title, picker = st.columns([5, 2])
    title.title("Credit Risk")

    labels = {entry["label"]: entry for entry in roles}
    chosen = picker.selectbox("Signed in as", list(labels))
    entry = labels[chosen]
    picker.caption(entry["description"])
    return entry["role"]


def _who(role: str, label: str) -> str | None:
    """Which member of staff is acting, since decisions record who took them."""
    try:
        staff = api.users(role=role)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return None

    if not staff:
        st.warning(
            f"No {label} accounts exist. Seed them with: "
            "`cd backend && python -m app.db.seed`"
        )
        return None

    by_name = {user["display_name"]: user["email"] for user in staff}
    chosen = st.sidebar.selectbox(label, list(by_name))
    st.sidebar.caption(by_name[chosen])
    return by_name[chosen]


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


main()
