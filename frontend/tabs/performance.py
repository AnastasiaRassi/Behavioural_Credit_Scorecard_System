"""Tab 1 · Performance. Headline metrics, confusion matrix, risk bands, deciles."""
import altair as alt
import pandas as pd
import streamlit as st

import loaders
from components import BAND_COLOURS, BAND_ICONS, explained_metric, thought
from explanations import TRUST_STABLE

HEADLINE = ["roc_auc", "f1_weighted", "balanced_accuracy", "gini"]
SECONDARY = ["f1_macro", "f1", "pr_auc", "ks", "brier", "mcc", "cohen_kappa"]


def _substitutions(metrics: dict, default_rate: float) -> dict:
    """Values formatted into the captions and popovers on this tab."""
    percentage = round(metrics["roc_auc"] * 100)
    return {
        "pct": percentage,
        "naive": percentage,
        "default_rate": round(default_rate * 100, 1),
        "default_rate_frac": f"{default_rate:.3f}",
        "recall": round(metrics["recall"] * 100),
        "specificity": round(metrics["specificity"] * 100),
    }


def render() -> None:
    settings = loaders.config()
    metrics = settings["test_metrics"]
    default_rate = settings["test_default_rate"]
    substitutions = _substitutions(metrics, default_rate)

    st.caption(
        f"All figures are on the held-out test split: {settings['test_rows']:,} "
        f"customers the model never saw, {default_rate:.1%} of whom defaulted. "
        f"Decision threshold {settings['threshold_f1_optimal']:.4f}."
    )

    for column, key in zip(st.columns(4), HEADLINE):
        with column:
            explained_metric(
                key, f"{metrics[key]:.3f}",
                caption_args=substitutions, deep_args=substitutions,
                trust=TRUST_STABLE,
            )

    st.divider()

    with st.expander("More metrics"):
        for start in range(0, len(SECONDARY), 4):
            for column, key in zip(st.columns(4), SECONDARY[start:start + 4]):
                with column:
                    explained_metric(
                        key, f"{metrics[key]:.3f}",
                        caption_args=substitutions, deep_args=substitutions,
                        trust=TRUST_STABLE,
                    )

    st.divider()
    left, right = st.columns(2)
    with left:
        _confusion_matrix(settings["confusion_matrix"])
    with right:
        _risk_bands(settings["band_report"])

    st.divider()
    _deciles(settings["deciles"], default_rate)


def _confusion_matrix(counts: dict) -> None:
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]

    header, icon = st.columns([6, 1])
    header.subheader("Confusion matrix")
    with icon:
        thought("confusion_matrix", args={"fn": f"{fn:,}", "fp": f"{fp:,}"})
    st.caption("At the live threshold, with counts and row percentages.")

    frame = pd.DataFrame(
        [
            {"actual": "pays", "predicted": "pays", "count": tn,
             "note": f"{tn / (tn + fp):.0%} correctly cleared"},
            {"actual": "pays", "predicted": "defaults", "count": fp,
             "note": "wrongly flagged"},
            {"actual": "defaults", "predicted": "pays", "count": fn,
             "note": "missed defaults"},
            {"actual": "defaults", "predicted": "defaults", "count": tp,
             "note": f"{tp / (tp + fn):.0%} caught"},
        ]
    )
    order = ["pays", "defaults"]
    base = alt.Chart(frame).encode(
        x=alt.X("predicted:N", sort=order, title="predicted"),
        y=alt.Y("actual:N", sort=order, title="actual"),
    )
    chart = (
        base.mark_rect().encode(
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None)
        )
        + base.mark_text(fontSize=13, lineBreak="\n").encode(
            text=alt.Text("label:N"),
            color=alt.condition(
                alt.datum.count > (tn + fp + fn + tp) / 4,
                alt.value("white"), alt.value("black"),
            ),
        ).transform_calculate(label="format(datum.count, ',') + '\\n' + datum.note")
    ).properties(height=260)
    st.altair_chart(chart, width="stretch")


def _risk_bands(bands: dict) -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Risk bands")
    with icon:
        thought("bands")
    st.caption("The table a loan officer actually acts on.")

    order = [band for band in ("Low", "Medium", "High") if band in bands]
    total = sum(bands[band]["customers"] for band in order)
    frame = pd.DataFrame(
        [
            {
                "band": band,
                "rate": bands[band]["actual_default_rate"],
                "share": bands[band]["customers"] / total,
            }
            for band in order
        ]
    )

    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("rate:Q", title="actual default rate", axis=alt.Axis(format="%")),
            y=alt.Y("band:N", sort=order, title=None),
            color=alt.Color(
                "band:N", sort=order, legend=None,
                scale=alt.Scale(
                    domain=order, range=[BAND_COLOURS[band] for band in order]
                ),
            ),
            tooltip=[
                alt.Tooltip("band:N"),
                alt.Tooltip("rate:Q", format=".1%", title="default rate"),
                alt.Tooltip("share:Q", format=".1%", title="share of customers"),
            ],
        )
        .properties(height=200)
    )
    st.altair_chart(chart, width="stretch")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Band": f"{BAND_ICONS[band]} {band}",
                    "Share of customers": f"{bands[band]['customers'] / total:.1%}",
                    "Actual default rate": f"{bands[band]['actual_default_rate']:.1%}",
                }
                for band in order
            ]
        ),
        hide_index=True, width="stretch",
    )


def _deciles(deciles: list[dict], default_rate: float) -> None:
    rates = [decile["actual_default_rate"] for decile in deciles]
    lowest, highest = min(rates), max(rates)

    header, icon = st.columns([6, 1])
    header.subheader("Risk deciles")
    with icon:
        thought(
            "deciles",
            args={
                "average": round(default_rate * 100, 1),
                "lowest": round(lowest * 100, 1),
                "highest": round(highest * 100, 1),
                "spread": round(highest / lowest, 1) if lowest else "—",
            },
        )
    st.caption(
        "Customers sorted by predicted risk into ten equal groups, safest first. "
        "The dashed line is the overall average."
    )

    frame = pd.DataFrame(
        [
            {
                "decile": f"decile {decile['decile']}",
                "order": decile["decile"],
                "rate": decile["actual_default_rate"],
                "customers": decile["customers"],
            }
            for decile in deciles
        ]
    )
    bars = (
        alt.Chart(frame)
        .mark_bar(color="#3b6fb5")
        .encode(
            x=alt.X("rate:Q", title="actual default rate", axis=alt.Axis(format="%")),
            y=alt.Y("decile:N", sort=alt.EncodingSortField("order"), title=None),
            tooltip=[
                alt.Tooltip("decile:N"),
                alt.Tooltip("rate:Q", format=".1%", title="default rate"),
                alt.Tooltip("customers:Q", format=",", title="customers"),
            ],
        )
    )
    average = (
        alt.Chart(pd.DataFrame({"rate": [default_rate]}))
        .mark_rule(strokeDash=[6, 4], color="#888")
        .encode(x="rate:Q")
    )
    st.altair_chart((bars + average).properties(height=420), width="stretch")
    st.caption(f"Overall average {default_rate:.1%}.")
