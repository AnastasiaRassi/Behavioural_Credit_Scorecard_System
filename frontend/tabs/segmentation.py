"""Segments tab. How customers behave, separate from the default model.

The segment is never a model input; this is a lens on the population.
"""
import pandas as pd
import streamlit as st

import api


def render() -> None:
    st.subheader("Behavioural segments")
    try:
        data = api.segments()
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    st.caption(
        f"k-means, k = {data['k']}, chosen by {data['chosen_by']}. "
        f"Fitted on the {data['fitted_on']}; profiles from the test split. "
        "Not an input to the default model."
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Segment": segment["name"],
                    "Customers": segment["customers"],
                    "Share": f"{segment['share']:.0%}",
                    "Default rate": f"{segment['default_rate']:.1%}",
                    "Distinctive": ", ".join(segment["distinctive"]),
                }
                for segment in data["segments"]
            ]
        ),
        hide_index=True, width="stretch",
    )

    scores = pd.Series(data["silhouette_scores"], name="Silhouette")
    scores.index.name = "k"
    st.markdown("**Silhouette score by k**")
    st.bar_chart(scores)

    with st.expander("Features used"):
        st.write(", ".join(data["features"]))
