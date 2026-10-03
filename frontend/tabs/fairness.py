"""Tab 2 · Fairness. Disparate impact, equalized odds, per-group detail.

The equalized-odds table carries a prominent instability warning. Those gaps come
from a single split and move substantially across splits; disparate impact does not.
Marking that difference is the point of the tab.
"""
import altair as alt
import pandas as pd
import streamlit as st

import loaders
from components import thought
from explanations import DEEP, TRUST_STABLE, TRUST_UNSTABLE

# Measured at training: what excluding the protected attributes costs in AUC.
EXCLUSION_AUC_COST = 0.0007


def render() -> None:
    audit = loaders.fairness()
    _verdict(audit)
    st.divider()
    _disparate_impact(audit)
    st.divider()
    _equalized_odds(audit)
    st.divider()
    _per_group(audit)


def _verdict(audit: dict) -> None:
    total = len(audit)
    passed = sum(1 for values in audit.values() if values["passes_four_fifths"])

    if passed == total:
        st.success(
            f"**ALL {total} PROTECTED ATTRIBUTES PASS THE FOUR-FIFTHS RULE**\n\n"
            f"Protected attributes are excluded from model inputs. "
            f"Cost of excluding them: {EXCLUSION_AUC_COST} AUC."
        )
    else:
        failing = [name for name, v in audit.items() if not v["passes_four_fifths"]]
        st.error(
            f"**{total - passed} OF {total} ATTRIBUTES FAIL THE FOUR-FIFTHS RULE**\n\n"
            f"Below {loaders.FOUR_FIFTHS:.2f}: {', '.join(failing)}."
        )
    thought("fairness_verdict")


def _disparate_impact(audit: dict) -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Disparate impact")
    with icon:
        thought("disparate_impact")
    st.caption(
        "Approval rate of the least-approved group divided by the most-approved."
    )

    frame = pd.DataFrame(
        [
            {
                "attribute": name,
                "ratio": values["disparate_impact_ratio"],
                "passes": bool(values["passes_four_fifths"]),
            }
            for name, values in audit.items()
        ]
    )

    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                "ratio:Q", title="disparate impact ratio",
                scale=alt.Scale(domain=[0, 1.05]),
            ),
            y=alt.Y("attribute:N", sort="-x", title=None),
            color=alt.Color(
                "passes:N", legend=None,
                scale=alt.Scale(domain=[True, False], range=["#2e9e5b", "#cc3b2f"]),
            ),
            tooltip=[alt.Tooltip("attribute:N"), alt.Tooltip("ratio:Q", format=".3f")],
        )
    )
    line = (
        alt.Chart(pd.DataFrame({"threshold": [loaders.FOUR_FIFTHS]}))
        .mark_rule(strokeDash=[6, 4], color="#cc3b2f")
        .encode(x="threshold:Q")
    )
    st.altair_chart((bars + line).properties(height=240), width="stretch")
    st.caption(f"The dashed line is the {loaders.FOUR_FIFTHS:.2f} four-fifths threshold.")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Attribute": name,
                    "DIR": f"{values['disparate_impact_ratio']:.3f}",
                    "Status": "✅ Pass" if values["passes_four_fifths"] else "❌ Fail",
                    "Trust": TRUST_STABLE,
                }
                for name, values in audit.items()
            ]
        ),
        hide_index=True, width="stretch",
    )

    closest = min(audit, key=lambda name: audit[name]["disparate_impact_ratio"])
    st.caption(
        f"{closest} at {audit[closest]['disparate_impact_ratio']:.3f} is closest to "
        "the line and worth watching."
    )


def _equalized_odds(audit: dict) -> None:
    st.subheader("Equalized odds")
    st.warning(DEEP["equalized_odds_warning"])

    def show(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "—"

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Attribute": name,
                    "TPR gap": show(values["equalized_odds_TPR_gap"]),
                    "FPR gap": show(values["equalized_odds_FPR_gap"]),
                    "AUC spread": show(values["auc_spread"]),
                    "Trust": TRUST_UNSTABLE,
                }
                for name, values in audit.items()
            ]
        ),
        hide_index=True, width="stretch",
    )

    for column, (key, label) in zip(
        st.columns(4),
        [
            ("tpr_gap", "💭 TPR gap"),
            ("fpr_gap", "💭 FPR gap"),
            ("auc_spread", "💭 AUC spread"),
            ("equalized_odds_why", "💭 Why it matters"),
        ],
    ):
        with column:
            thought(key, label=label)


def _per_group(audit: dict) -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Per-group detail")
    with icon:
        thought("per_group")

    for attribute, values in audit.items():
        groups = [group for group in values["groups"] if group["reliable"]]
        hidden = values["hidden_groups"]

        with st.expander(f"{attribute} · {len(groups)} groups"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Group": group["group"],
                            "Customers": f"{group['customers']:,}",
                            "Defaults": f"{group['defaults']:,}",
                            "Default rate": f"{group['actual_default_rate']:.1%}",
                            "Approval rate": f"{group['approval_rate']:.1%}",
                            "TPR": (
                                f"{group['tpr']:.3f}" if group["tpr"] is not None else "—"
                            ),
                            "FPR": (
                                f"{group['fpr']:.3f}" if group["fpr"] is not None else "—"
                            ),
                            "AUC": (
                                f"{group['auc']:.3f}" if group["auc"] is not None else "—"
                            ),
                        }
                        for group in groups
                    ]
                ),
                hide_index=True, width="stretch",
            )
            if hidden:
                st.caption(
                    f"Hidden as too small to read: {', '.join(hidden)}. A group needs "
                    "at least 50 members and 10 defaults."
                )
