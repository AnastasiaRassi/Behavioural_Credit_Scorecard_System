"""Tab 3 · Explainability. Global importance, and a single-prediction explainer.

Global importance is precomputed. The single-prediction panel is the one place the
dashboard calls the model live, because the engineer chooses the inputs.
"""
import altair as alt
import pandas as pd
import streamlit as st

import loaders
from components import BAND_COLOURS, BAND_ICONS, thought
from explanations import DEEP

TOP_N = 20
SHOWN_REASONS = 8
PAY_STATUS_OPTIONS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def render() -> None:
    _global_importance()
    st.divider()
    _single_prediction()


def _global_importance() -> None:
    from app.ml.explainability import feature_meaning

    header, icon = st.columns([6, 1])
    header.subheader("Global feature importance")
    with icon:
        thought("feature_importance")
    st.caption(
        "Permutation importance: each feature is scrambled in turn and we measure "
        "how far test AUC drops. Error bars are the spread over five repeats."
    )

    rows = loaders.importance()
    top, rest = rows[:TOP_N], rows[TOP_N:]
    frame = pd.DataFrame(top).assign(
        meaning=lambda f: f["feature"].map(feature_meaning),
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
            "Plain meaning": feature_meaning(row["feature"]),
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
    st.caption("Poke the model. Change a customer's history and watch the score move.")

    scorer, explainer = loaders.explainer()
    customer = _input_panel()

    if st.button("Score customer", type="primary"):
        _result(scorer, explainer, customer)


def _input_panel() -> dict:
    """Editable copy of the template customer. Defaults are a stressed profile."""
    from app.ml.feature_engineering import describe_pay_status
    from app.ml.scorer import example_customer

    customer = example_customer()

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
                index=PAY_STATUS_OPTIONS.index(2), format_func=describe_pay_status,
            )
            customer["PAY_2"] = st.selectbox(
                "Previous month", PAY_STATUS_OPTIONS,
                index=PAY_STATUS_OPTIONS.index(2), format_func=describe_pay_status,
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
            "The earlier months inherit the two you set, so the six-month profile "
            "stays internally consistent. Age is collected because the fairness "
            "audit needs it, and is dropped before the model sees the customer."
        )

    # Carry the recent pattern backwards rather than leaving a healthy tail behind.
    for month in range(3, 7):
        customer[f"PAY_{month}"] = customer["PAY_2"]
    for month in range(2, 7):
        customer[f"BILL_AMT{month}"] = int(
            customer["BILL_AMT1"] * (1 - 0.02 * (month - 1))
        )
        customer[f"PAY_AMT{month}"] = customer["PAY_AMT1"]

    return customer


def _result(scorer, explainer, customer: dict) -> None:
    result = scorer.score_one(customer)
    band = result.risk_band

    st.markdown(
        f"### {result.probability:.1%} &nbsp; "
        f"<span style='color:{BAND_COLOURS[band]}'>"
        f"{BAND_ICONS[band]} {band.upper()} RISK</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"This band defaults {result.band_default_rate:.1%} of the time on held-out "
        f"customers. Flagged at the production threshold: "
        f"{'yes' if result.flagged else 'no'} (cut-off {result.threshold:.4f})."
    )

    reasons, _ = explainer.reasons(customer, limit=SHOWN_REASONS)
    st.markdown("**Why:**")
    for reason in reasons:
        arrow = "▲" if reason.raises_risk else "▼"
        colour = "#cc3b2f" if reason.raises_risk else "#2e9e5b"
        st.markdown(
            f"<span style='color:{colour}'>{arrow}</span> {reason.description} "
            f"&nbsp; <code>{reason.effect:+.3f}</code>",
            unsafe_allow_html=True,
        )

    st.caption(
        "Effects are in log-odds and sum to the model's raw score, which calibration "
        "then maps to the probability above. The ordering and relative sizes are "
        "exact; only the final percentage comes from calibration."
    )
