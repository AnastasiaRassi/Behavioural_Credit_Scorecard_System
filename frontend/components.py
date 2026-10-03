"""Shared widgets: the two-level explanation pattern used on every metric.

Level one is a caption always visible under the number. Level two is a 💭 popover
holding a longer plain-language explanation. Nobody should have to already know
what a Brier score is.
"""
import streamlit as st

from explanations import CAPTIONS, DEEP, TRUST_STABLE


def explained_metric(
    key: str,
    value: str,
    *,
    caption_args: dict | None = None,
    deep_args: dict | None = None,
    trust: str | None = TRUST_STABLE,
    label: str | None = None,
) -> None:
    """A metric tile with its caption, trust marker and 💭 popover."""
    caption = CAPTIONS.get(key, "").format(**(caption_args or {}))
    deep = DEEP.get(key, "").format(**(deep_args or {}))

    st.metric(label or _label(key), value)
    if caption:
        st.caption(caption)
    if trust:
        st.caption(trust)
    if deep.strip():
        with st.popover("💭", width="content"):
            st.markdown(deep)


def thought(key: str, *, args: dict | None = None, label: str = "💭") -> None:
    """A standalone 💭 popover for a chart or table."""
    body = DEEP.get(key, "").format(**(args or {}))
    if not body.strip():
        return
    with st.popover(label, width="content"):
        st.markdown(body)


LABELS = {
    "roc_auc": "ROC-AUC",
    "f1_weighted": "Weighted F1",
    "balanced_accuracy": "Balanced accuracy",
    "gini": "Gini",
    "f1_macro": "Macro F1",
    "f1": "F1, default class",
    "pr_auc": "PR-AUC",
    "ks": "KS statistic",
    "brier": "Brier score",
    "mcc": "MCC",
    "cohen_kappa": "Cohen kappa",
}


def _label(key: str) -> str:
    return LABELS.get(key, key.replace("_", " ").title())


# Band styling, shared by the performance tab and the officer view.
BAND_COLOURS = {"Low": "#2e9e5b", "Medium": "#d9a017", "High": "#cc3b2f"}
BAND_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
