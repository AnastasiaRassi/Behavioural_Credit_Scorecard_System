"""Tab 5 · Registry. Model versions, newest first.

Promotion is gated on the fairness audit. The gate is enforced here for the
operator's benefit and must be enforced again in the API, since a UI check is not
a control.
"""
import pandas as pd
import streamlit as st

import loaders
from components import thought

STATUS_ICONS = {"live": "🟢 Live", "archived": "⚪ Archived", "candidate": "🔵 Candidate"}


def render() -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Model registry")
    with icon:
        thought("registry")

    rows = loaders.registry()
    settings = loaders.config()

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Version": row["version"],
                    "Trained": row["trained"],
                    "AUC": f"{row['roc_auc']:.3f}",
                    "F1": f"{row['f1']:.3f}",
                    "Fairness": (
                        f"✅ {row['fairness_passed']}/{row['fairness_total']}"
                        if row["fairness_passed"] == row["fairness_total"]
                        else f"❌ {row['fairness_passed']}/{row['fairness_total']}"
                    ),
                    "Status": STATUS_ICONS.get(row["status"], row["status"]),
                }
                for row in rows
            ]
        ),
        hide_index=True, width="stretch",
    )

    if len(rows) == 1:
        st.info(
            "Only one version has been trained, so there is nothing to compare or "
            "promote yet. The table fills as models accumulate."
        )

    _detail(rows, settings)
    _actions(rows)


def _detail(rows: list[dict], settings: dict) -> None:
    with st.expander("Full metrics for the selected version"):
        version = st.selectbox("Version", [row["version"] for row in rows])
        metrics = settings["test_metrics"]

        st.caption(
            f"{version} · trained {settings['trained']} · "
            f"{settings['training_rows']:,} training rows · "
            f"{settings['n_features']} features · "
            f"seeds {'/'.join(str(seed) for seed in settings['seeds'])}"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": key.replace("_", " "), "Value": f"{value:.4f}"}
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                ]
            ),
            hide_index=True, width="stretch",
        )


def _actions(rows: list[dict]) -> None:
    st.subheader("Actions")
    candidates = [row for row in rows if row["status"] != "live"]

    if not candidates:
        st.caption("No candidate versions to promote or archive.")
        return

    choice = st.selectbox("Candidate", [row["version"] for row in candidates])
    selected = next(row for row in candidates if row["version"] == choice)
    eligible = selected["fairness_passed"] == selected["fairness_total"]

    if not eligible:
        st.error(
            f"{choice} cannot be promoted: its fairness audit failed "
            f"({selected['fairness_passed']}/{selected['fairness_total']} passed)."
        )

    promote, archive = st.columns(2)
    promote.button(
        f"Promote to live{'' if eligible else ' (blocked)'}",
        disabled=not eligible, width="stretch",
        help="A version cannot be promoted unless its fairness audit passed.",
    )
    archive.button("Archive", width="stretch")

    st.caption(
        "Promotion and archiving are not yet wired to a backing store, so these "
        "buttons do not change the live model."
    )
