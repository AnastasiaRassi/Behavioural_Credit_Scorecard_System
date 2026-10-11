"""Tab 3 · Explainability. Global importance, and a single-prediction explainer.

Global importance is precomputed. The single-prediction panel is the one place the
dashboard calls the model live, because the engineer chooses the inputs.
"""
import altair as alt
import pandas as pd
import streamlit as st

import api
from components import BAND_COLOURS, BAND_ICONS, thought
from explanations import DEEP

TOP_N = 20
SHOWN_REASONS = 8
PAY_STATUS_OPTIONS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]

# Mirrors describe_pay_status() in the backend. Duplicated only because this
# panel builds a hypothetical customer before any request is made.
PAY_STATUS_READINGS = {
    -2: "no balance", -1: "paid in full", 0: "revolving credit",
    1: "1 month in arrears",
}

# The API validates the full set of columns, so the panel starts from a complete
# customer and overwrites the fields on screen.
EXAMPLE_CUSTOMER = {
    "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 40000, "BILL_AMT2": 38000, "BILL_AMT3": 36000,
    "BILL_AMT4": 34000, "BILL_AMT5": 32000, "BILL_AMT6": 30000,
    "PAY_AMT1": 3000, "PAY_AMT2": 3000, "PAY_AMT3": 3000,
    "PAY_AMT4": 3000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
}


def render() -> None:
    _global_importance()
    st.divider()
    _single_prediction()


def _global_importance() -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Global feature importance")
    with icon:
        thought("feature_importance")
    st.caption(
        "Permutation importance: each feature is scrambled in turn and we measure "
        "how far test AUC drops. Error bars are the spread over five repeats."
    )

    rows = api.importance()
    top, rest = rows[:TOP_N], rows[TOP_N:]
    frame = pd.DataFrame(top).assign(
        meaning=lambda f: f["feature"],
        low=lambda f: f["auc_drop"] - f["std"],
        high=lambda f: f["auc_drop"] + f["std"],
    )

    order = alt.EncodingSortField("auc_drop", order="descending")
    bars = (
        alt.Chart(frame)
        .mark_bar(color="#3b6fb5")
        .encode(
            x=alt.X("auc_drop:Q", title="AUC drop when shuffled"),
            y=alt.Y("feature:N", sort=order, title=None),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("meaning:N", title="meaning"),
                alt.Tooltip("auc_drop:Q", format=".4f", title="AUC drop"),
                alt.Tooltip("std:Q", format=".4f", title="±"),
            ],
        )
    )
    errors = (
        alt.Chart(frame)
        .mark_rule(color="#999")
        .encode(x="low:Q", x2="high:Q", y=alt.Y("feature:N", sort=order))
    )
    st.altair_chart((bars + errors).properties(height=520), width="stretch")

    st.warning(DEEP["feature_importance_caveat"])

    table = [
        {
            "Feature": row["feature"],
            "AUC drop when shuffled": f"{row['auc_drop']:.4f}",
            "±": f"{row['std']:.4f}",
            "Plain meaning": row["feature"],
        }
        for row in top[:5]
    ]
    if rest:
        table.append(
            {
                "Feature": f"{len(rest)} others combined",
                "AUC drop when shuffled": f"~{sum(r['auc_drop'] for r in rest):.4f}",
                "±": "—",
                "Plain meaning": "everything else",
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


def _single_prediction() -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Single prediction explainer")
    with icon:
        thought("single_prediction")
    st.caption(
        "Poke the model. Nothing here is recorded: an experiment is not a "
        "decision, and writing every slider movement into the audit trail would "
        "bury the real ones."
    )

    customer = _input_panel()
    if not st.button("Score customer", type="primary"):
        return

    try:
        result = api.explain(customer)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    _result(result)


def _input_panel() -> dict:
    """Editable copy of the template customer. Defaults are a stressed profile."""
    customer = dict(EXAMPLE_CUSTOMER)

    with st.expander("Test a customer", expanded=True):
        first, second, third = st.columns(3)

        with first:
            customer["LIMIT_BAL"] = st.number_input(
                "Credit limit", min_value=10_000, max_value=1_000_000,
                value=200_000, step=10_000,
            )
            customer["AGE"] = st.number_input("Age", min_value=21, max_value=79, value=35)

        with second:
            customer["PAY_1"] = st.selectbox(
                "Last month's repayment status", PAY_STATUS_OPTIONS,
                index=PAY_STATUS_OPTIONS.index(2), format_func=_pay_status,
            )
            customer["PAY_2"] = st.selectbox(
                "Previous month", PAY_STATUS_OPTIONS,
                index=PAY_STATUS_OPTIONS.index(2), format_func=_pay_status,
            )

        with third:
            customer["BILL_AMT1"] = st.number_input(
                "Current balance", min_value=0, max_value=1_000_000,
                value=190_000, step=5_000,
            )
            customer["PAY_AMT1"] = st.number_input(
                "Last payment made", min_value=0, max_value=1_000_000,
                value=0, step=1_000,
            )

        st.caption(
            "Earlier months inherit the two you set, so the six-month profile "
            "stays internally consistent. Age is accepted because the audit needs "
            "it, and is dropped before the model sees the customer."
        )

    for month in range(3, 7):
        customer[f"PAY_{month}"] = customer["PAY_2"]
    for month in range(2, 7):
        customer[f"BILL_AMT{month}"] = int(
            customer["BILL_AMT1"] * (1 - 0.02 * (month - 1))
        )
        customer[f"PAY_AMT{month}"] = customer["PAY_AMT1"]

    return customer


def _pay_status(value: int) -> str:
    return PAY_STATUS_READINGS.get(value, f"{value} months in arrears")


def _result(result: dict) -> None:
    band = result["risk_band"]
    st.markdown(
        f"### {result['probability']:.1%} &nbsp; "
        f"<span style='color:{BAND_COLOURS[band]}'>"
        f"{BAND_ICONS[band]} {band.upper()} RISK</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"This band defaults {result['band_default_rate']:.1%} of the time on "
        f"held-out customers. Flagged at the production threshold: "
        f"{'yes' if result['flagged'] else 'no'} (cut-off {result['threshold']:.4f})."
    )

    st.markdown("**Why:**")
    for reason in result["reasons"][:SHOWN_REASONS]:
        raises = reason["raises_risk"]
        arrow = "▲" if raises else "▼"
        colour = "#cc3b2f" if raises else "#2e9e5b"
        st.markdown(
            f"<span style='color:{colour}'>{arrow}</span> {reason['description']} "
            f"&nbsp; <code>{reason['effect']:+.3f}</code>",
            unsafe_allow_html=True,
        )

    st.caption(
        "Effects are in log-odds and sum to the model's raw score, which "
        "calibration maps to the probability above. Ordering and relative sizes "
        "are exact; only the final percentage comes from calibration."
    )
