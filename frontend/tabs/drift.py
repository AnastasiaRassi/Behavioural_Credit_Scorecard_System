"""Tab 6 · Drift. A design placeholder, by the spec's own instruction.

PSI needs repeated scoring over time. The dataset is a single October 2005 snapshot,
so there is no second period to compare against. Showing an invented heatmap here
would be the one dishonest panel in the dashboard, so the tab states the gap instead.
"""
import streamlit as st

import loaders
from components import thought

WATCH = 0.10
SIGNIFICANT = 0.25


def render() -> None:
    header, icon = st.columns([6, 1])
    header.subheader("Population stability")
    with icon:
        thought("drift")

    st.info(
        "**Design placeholder — no data source exists.**\n\n"
        "PSI compares the customers arriving now against the customers the model was "
        "trained on. That needs at least two time periods of scored customers. The "
        "dataset is a single snapshot of one month (October 2005), so there is no "
        "second period to compare against and every number on this tab would be "
        "fabricated."
    )

    st.markdown(
        f"""
**What this tab will show once the system scores customers over time**

A heatmap of features by week, with each cell holding that feature's PSI against the
training distribution. Thresholds at **{WATCH:.2f}** (watch) and **{SIGNIFICANT:.2f}**
(significant), which are the conventional bands.

**What it needs first**

- A scoring log: every customer scored, with their features and the date
- At least two complete periods, ideally weekly, to form a comparison
- A stored reference distribution per feature, taken from the training split

The reference half of that already exists — the training split is deterministic
(`build_splits(seed=42)`), so the baseline distribution is reproducible today. Only
the incoming side is missing.
"""
    )

    settings = loaders.config()
    st.caption(
        f"For reference, the model was trained on {settings['training_rows']:,} "
        f"customers across {settings['n_features']} features, which is the "
        f"distribution any future "
        "PSI would be measured against."
    )
