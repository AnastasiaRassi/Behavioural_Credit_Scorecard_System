"""Tab 4 · Threshold and bands. Interactive, and the most useful tab for an engineer.

The slider reads the precomputed sweep rather than rescoring, so dragging it is
instant and the numbers still come from the real test split.
"""
import altair as alt
import pandas as pd
import streamlit as st

import loaders
from components import thought

REFERENCE_THRESHOLDS = [0.20, 0.35, 0.50, 0.60]
DEFAULT_COST_RATIO = 5


def render() -> None:
    settings = loaders.config()
    production = settings["threshold_f1_optimal"]
    sweep = loaders.sweep()

    header, icon = st.columns([6, 1])
    header.subheader("Decision threshold")
    with icon:
        thought("threshold")

    live = st.slider(
        "Decision threshold",
        min_value=float(min(row["threshold"] for row in sweep)),
        max_value=float(max(row["threshold"] for row in sweep)),
        value=float(production), step=0.005, format="%.4f",
        help="The cut-off where a score becomes a decision.",
        label_visibility="collapsed",
    )
    row = loaders.nearest_sweep_row(live)
    moved = live - production

    st.caption(
        f"Saved production threshold is **{production:.4f}**. "
        + ("You are on it." if abs(moved) < 1e-9 else f"You have moved {moved:+.4f} from it.")
    )

    for column, (label, value, explanation) in zip(
        st.columns(4),
        [
            ("Flagged", f"{row['flagged_share']:.1%}", "Share of customers flagged."),
            ("Precision", f"{row['precision']:.3f}",
             "Of those flagged, the share who really default."),
            ("Recall", f"{row['recall']:.3f}", "Of all defaulters, the share we flag."),
            ("F1", f"{row['f1']:.3f}", "Harmonic mean of precision and recall."),
        ],
    ):
        column.metric(label, value, help=explanation)

    st.caption(
        f"At this threshold: **{row['tp']:,}** defaulters caught, "
        f"**{row['fn']:,}** missed, **{row['fp']:,}** good customers wrongly "
        f"flagged, **{row['tn']:,}** correctly cleared."
    )

    st.divider()
    _tradeoff(sweep, live, production)
    st.divider()
    _reference_table(row, production)
    st.divider()
    _cost_curve(production)
    st.divider()
    _calibration(settings)


def _markers(live: float, production: float) -> alt.Chart:
    frame = pd.DataFrame(
        [{"threshold": live, "marker": "live"},
         {"threshold": production, "marker": "production"}]
    )
    return (
        alt.Chart(frame)
        .mark_rule()
        .encode(
            x="threshold:Q",
            color=alt.Color(
                "marker:N",
                scale=alt.Scale(domain=["live", "production"], range=["#cc3b2f", "#888"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "marker:N",
                scale=alt.Scale(domain=["live", "production"], range=[[1, 0], [4, 4]]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("marker:N"), alt.Tooltip("threshold:Q", format=".4f")],
        )
    )


def _tradeoff(sweep: list[dict], live: float, production: float) -> None:
    st.subheader("Precision against recall")

    frame = pd.DataFrame(sweep)[["threshold", "precision", "recall", "f1"]]
    melted = frame.melt("threshold", var_name="metric", value_name="score")

    lines = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="threshold"),
            y=alt.Y("score:Q", title="score"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["precision", "recall", "f1"],
                    range=["#3b6fb5", "#d9a017", "#2e9e5b"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("metric:N"),
                alt.Tooltip("score:Q", format=".3f"),
            ],
        )
    )
    st.altair_chart(
        (lines + _markers(live, production)).properties(height=320),
        width="stretch",
    )


def _reference_table(live_row: dict, production: float) -> None:
    st.subheader("Reference points")

    rows = [(loaders.nearest_sweep_row(target), False) for target in REFERENCE_THRESHOLDS]
    rows.append((live_row, True))
    rows.sort(key=lambda item: item[0]["threshold"])

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Threshold": (
                        f"{row['threshold']:.3f}"
                        + (" (live)" if is_live else "")
                        + (" ← production" if abs(row["threshold"] - production) < 0.003 else "")
                    ),
                    "Flagged": f"{row['flagged_share']:.0%}",
                    "Precision": f"{row['precision']:.3f}",
                    "Recall": f"{row['recall']:.3f}",
                    "F1": f"{row['f1']:.3f}",
                }
                for row, is_live in rows
            ]
        ),
        hide_index=True, width="stretch",
    )


def _cost_curve(production: float) -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Cost curve")
    with icon:
        thought("cost_curve")

    curves = loaders.cost_curves()
    ratio = st.slider(
        "Cost of a missed default relative to a wrongly refused customer",
        min_value=min(int(key) for key in curves),
        max_value=max(int(key) for key in curves),
        value=DEFAULT_COST_RATIO, step=1, format="%d : 1",
    )
    entry = curves[str(ratio)]
    best = entry["best_threshold"]

    line = (
        alt.Chart(pd.DataFrame(entry["curve"]))
        .mark_line(color="#3b6fb5")
        .encode(
            x=alt.X("threshold:Q", title="threshold"),
            y=alt.Y(
                "cost:Q",
                title="cost (units of one wrongly refused customer)",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("cost:Q", format=",.0f"),
            ],
        )
    )
    optimum = (
        alt.Chart(pd.DataFrame({"threshold": [best]}))
        .mark_rule(color="#2e9e5b")
        .encode(x="threshold:Q")
    )
    saved = (
        alt.Chart(pd.DataFrame({"threshold": [production]}))
        .mark_rule(color="#888", strokeDash=[4, 4])
        .encode(x="threshold:Q")
    )
    st.altair_chart(
        (line + optimum + saved).properties(height=320), width="stretch"
    )
    st.caption(
        f"At {ratio}:1 the cost-minimising threshold is **{best:.3f}** (green), "
        f"against the production threshold of {production:.4f} (grey dashes)."
    )


def _calibration(settings: dict) -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Calibration")
    with icon:
        thought("calibration")

    frame = pd.DataFrame(settings["calibration"])
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[6, 4], color="#888")
        .encode(x="x:Q", y="y:Q")
    )
    observed = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=70, color="#3b6fb5"), color="#3b6fb5")
        .encode(
            x=alt.X("mean_predicted:Q", title="predicted probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed_rate:Q", title="observed default rate",
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("mean_predicted:Q", format=".3f", title="predicted"),
                alt.Tooltip("observed_rate:Q", format=".3f", title="observed"),
                alt.Tooltip("customers:Q", format=",", title="customers"),
            ],
        )
    )
    st.altair_chart((diagonal + observed).properties(height=360), width="stretch")
    st.caption(
        f"Brier score {settings['test_metrics']['brier']:.3f}. "
        "Lower is better, 0 is perfect. The dashed line is perfect calibration."
    )
