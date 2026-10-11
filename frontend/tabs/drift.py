"""Tab 6 · Monitoring. Has the population moved, and is the model still right.

No longer a placeholder: both panels come from the API, which reports what the
stored data can actually support and refuses to invent a number when there are
too few rows. Every prediction an officer runs adds to what this can measure.
"""
import altair as alt
import pandas as pd
import streamlit as st

import api
import theme
from components import thought

WATCH = 0.10
SIGNIFICANT = 0.25


def render() -> None:
    try:
        drift = api.drift()
        performance = api.performance()
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    _population(drift)
    st.divider()
    _performance(performance)


def _population(drift: dict) -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Population stability")
    with icon:
        thought("drift")

    if not drift.get("sufficient"):
        st.info(
            f"**Not enough scored customers yet.** {drift.get('reason', '')}\n\n"
            "PSI compares the customers arriving now against those the model was "
            "trained on, so it needs scored customers to accumulate. Every "
            "prediction an officer runs adds one."
        )
        return

    frame = pd.DataFrame(drift["features"])
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("psi:Q", title="population stability index"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.psi >= SIGNIFICANT,
                alt.value(theme.HIGH),
                alt.condition(
                    alt.datum.psi >= WATCH, alt.value(theme.MEDIUM), alt.value(theme.LOW)
                ),
            ),
            tooltip=["feature:N", alt.Tooltip("psi:Q", format=".3f")],
        )
        .properties(height=min(26 * len(frame) + 40, 640))
    )
    st.altair_chart(chart, width="stretch", theme=None)
    st.caption(
        f"Green below {WATCH:.2f}, amber from {WATCH:.2f} (watch), "
        f"red from {SIGNIFICANT:.2f} (significant)."
    )


def _performance(performance: dict) -> None:
    st.subheader("Production performance")

    if not performance.get("sufficient"):
        st.info(
            f"**Not measurable yet.** {performance.get('reason', '')}\n\n"
            "Performance needs the real outcome, which arrives a month after the "
            "score. Until those labels exist, any figure here would be noise."
        )
        st.caption(
            "This is the honest limit of a single-snapshot dataset: it has no "
            "second month in which outcomes could be observed."
        )
        return

    for column, (label, key, fmt) in zip(
        st.columns(4),
        [
            ("Labelled predictions", "labelled_predictions", "{:,}"),
            ("ROC-AUC", "roc_auc", "{:.3f}"),
            ("F1, default class", "f1_default_class", "{:.3f}"),
            ("Brier", "brier", "{:.3f}"),
        ],
    ):
        column.metric(label, fmt.format(performance[key]))
