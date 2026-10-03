"""Credit Risk · Model Operations dashboard.

Run from the repo root:

    streamlit run frontend/app.py

Every number on screen is loaded from backend/artifacts. The dashboard never
retrains or re-evaluates, so the screen always agrees with what was actually
evaluated. Regenerate the artifacts from backend/ with:

    python -m app.ml.train
    python -m app.ml.dashboard_artifacts
"""
import sys
from pathlib import Path

import streamlit as st

# Allow `import loaders` and `import tabs.*` when Streamlit runs this by path.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import loaders  # noqa: E402
from tabs import drift, explain, fairness, performance, registry, threshold  # noqa: E402

VIEWS = ["ML Engineer", "Loan Officer"]

st.set_page_config(
    page_title="Credit Risk · Model Operations",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    title, picker = st.columns([5, 1])
    title.title("Credit Risk · Model Operations")
    view = picker.selectbox("View", VIEWS, label_visibility="collapsed")

    if view != "ML Engineer":
        st.info(
            "The loan officer view is not built yet. It has no specification in "
            "`docs/`, unlike this one. Switch back to **ML Engineer** above."
        )
        return

    _header()
    _tabs()


def _header() -> None:
    settings = loaders.config()
    colour, label, reason = loaders.health()

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


def _tabs() -> None:
    names = [
        "Performance", "Fairness", "Explainability",
        "Threshold", "Registry", "Drift",
    ]
    renderers = [
        performance.render, fairness.render, explain.render,
        threshold.render, registry.render, drift.render,
    ]

    for tab, render in zip(st.tabs(names), renderers):
        with tab:
            render()


main()
