"""Loan officer view: find a customer, read their account, predict, decide.

Search is the way in, not a leaderboard. A queue sorted by risk shows a thousand
high-risk customers and no officer works a thousand of anything; the actions on
offer here are account management, which starts from a customer rather than a
ranking. The queue stays as a second tab, framed as this month's batch triage.

Demographics never appear. They are dropped before scoring (conventions §10) and
putting them beside a score would invite the discrimination the audit rules out.
"""
import altair as alt
import pandas as pd
import streamlit as st

import api
from components import BAND_ICONS

ACTIONS = {
    "no_action": "No action",
    "monitor": "Monitor",
    "contact_customer": "Contact customer",
    "reduce_limit": "Reduce limit",
}
BANDS = ["High", "Medium", "Low"]
PAGE_SIZE = 25

# No model is certain, so the extremes are shown as bounds.
CERTAINTY_CAP = 0.99

STATUS_COLOURS = {"arrears": "#cc3b2f", "revolving": "#d9a017", "clear": "#2e9e5b"}


def render(officer_email: str) -> None:
    find, triage = st.tabs(["Find a customer", "This month's triage"])
    with find:
        _lookup(officer_email)
    with triage:
        _queue()


# --- finding a customer ---------------------------------------------------


def _lookup(officer_email: str) -> None:
    st.caption(
        "Look up the customer in front of you and check whether their next "
        "payment is at risk."
    )

    typed = st.text_input(
        "Customer", placeholder="Name or account number, e.g. Maya Haddad or 14861"
    )
    customer_id = _resolve(typed)
    if customer_id is None:
        return

    try:
        detail = api.customer_detail(customer_id)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    _profile(detail)
    st.divider()
    _history(detail)
    st.divider()
    _prediction(detail, officer_email)
    st.divider()
    _decision(detail, officer_email)


def _resolve(typed: str) -> str | None:
    """Turn what the officer typed into a customer id, forgivingly."""
    if not typed.strip():
        _suggestions()
        return None

    try:
        matches = api.search_customers(typed.strip())
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return None

    if not matches:
        st.warning(f"No customer matches {typed!r}.")
        return None
    if len(matches) == 1:
        return matches[0]["customer_id"]

    labels = {_label(match): match["customer_id"] for match in matches}
    return labels[st.selectbox(f"{len(matches)} customers match. Which one?", labels)]


def _label(match: dict) -> str:
    if match["full_name"]:
        return f"{match['full_name']} · {match['customer_id']}"
    return f"{match['customer_id']} (no name on file)"


def _suggestions() -> None:
    """Accounts to open without typing, so the screen is never a dead end."""
    try:
        known = api.users(role="cardholder")
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return
    if known:
        st.caption("Try: " + ", ".join(
            f"**{user['full_name']}** (`{user['display_name']}`)"
            for user in known if user.get("full_name")
        ))


# --- the customer page ----------------------------------------------------


def _profile(detail: dict) -> None:
    segment = detail.get("segment") or {}
    highlights = detail["highlights"]

    name = detail.get("full_name")
    st.subheader(name or f"Customer {detail['customer_id']}")
    st.caption(f"Account {detail['customer_id']}")
    if segment:
        st.caption(f"Behavioural segment: **{segment['name']}**")

    utilisation = highlights.get("utilisation_now")
    for column, (label, value, explanation) in zip(
        st.columns(5),
        [
            ("Credit limit", f"{detail['credit_limit']:,.0f}", "The agreed limit."),
            ("Balance", f"{detail['current_balance']:,.0f}", "Owed this month."),
            ("Last payment", f"{detail['last_payment']:,.0f}", "Most recent payment."),
            (
                "Utilisation",
                f"{utilisation:.0%}" if utilisation is not None else "-",
                "Balance as a share of the limit.",
            ),
            (
                "In arrears",
                f"{detail['months_in_arrears']} of 6",
                "Months behind in the last six.",
            ),
        ],
    ):
        column.metric(label, value, help=explanation)

    st.markdown(" &nbsp;·&nbsp; ".join(_notes(highlights)))


def _notes(highlights: dict) -> list[str]:
    """The short reads an officer wants before looking at anything else."""
    notes = []

    trajectory = highlights.get("trajectory")
    if trajectory:
        notes.append(f"Repayment behaviour is **{trajectory}**")

    missed = highlights.get("months_with_no_payment")
    if missed:
        notes.append(f"**{missed}** of six months with no payment")

    since = highlights.get("months_since_full_payment")
    notes.append(
        f"Last paid in full **{since} months ago**" if since is not None
        else "**Never** paid in full in the observed window"
    )
    return notes


def _history(detail: dict) -> None:
    st.markdown("**Last six months**")
    st.caption("Most recent first. This is the behaviour the model reads.")

    frame = pd.DataFrame(detail["history"])
    frame["status"] = frame["payment_status"].map(
        lambda value: "arrears" if value > 0 else
        ("revolving" if value == 0 else "clear")
    )

    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=list(frame["label"]), title=None),
            y=alt.Y("bill_amount:Q", title="bill"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=list(STATUS_COLOURS), range=list(STATUS_COLOURS.values())
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="month"),
                alt.Tooltip("payment_status_reading:N", title="status"),
                alt.Tooltip("bill_amount:Q", format=",.0f", title="bill"),
                alt.Tooltip("amount_paid:Q", format=",.0f", title="paid"),
            ],
        )
        .properties(height=200)
    )
    st.altair_chart(bars, width="stretch")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Month": entry["label"],
                    "Repayment status": entry["payment_status_reading"],
                    "Bill": f"{entry['bill_amount']:,.0f}",
                    "Paid": f"{entry['amount_paid']:,.0f}",
                    "Utilisation": (
                        f"{entry['utilisation']:.0%}"
                        if entry.get("utilisation") is not None else "-"
                    ),
                }
                for entry in detail["history"]
            ]
        ),
        hide_index=True, width="stretch",
    )


def _prediction(detail: dict, officer_email: str) -> None:
    st.markdown("**Next payment**")

    stored = detail.get("latest_prediction")
    state_key = f"prediction_{detail['customer_id']}"
    fresh = st.session_state.get(state_key)

    if st.button("Predict next payment", type="primary"):
        try:
            fresh = api.predict(detail["customer_id"], scored_by=officer_email)
        except api.ServiceUnavailable as exception:
            st.error(str(exception))
            return
        st.session_state[state_key] = fresh
        api.customer_detail.clear()

    if fresh is None and stored is None:
        band = detail["risk_band"]
        st.info(
            f"This account sits in the **{BAND_ICONS[band]} {band}** band in this "
            "month's batch. Run the prediction to see the risk and the reasons "
            "behind it."
        )
        return

    result = fresh or {
        "probability": stored["probability"],
        "risk_band": stored["risk_band"],
        "flagged": stored["flagged"],
        "threshold": stored["threshold"],
        "reasons": stored["explanation"]["reasons"],
    }
    _show_prediction(result, is_fresh=fresh is not None)


def _as_risk(probability: float) -> str:
    if probability >= CERTAINTY_CAP:
        return f"over {CERTAINTY_CAP:.0%}"
    if probability <= 1 - CERTAINTY_CAP:
        return f"under {1 - CERTAINTY_CAP:.0%}"
    return f"{probability:.0%}"


def _show_prediction(result: dict, is_fresh: bool) -> None:
    band = result["risk_band"]

    headline, verdict = st.columns([1, 2])
    headline.metric("Risk of missing the next payment", _as_risk(result["probability"]))
    verdict.markdown(
        f"### {BAND_ICONS[band]} {band} risk\n"
        f"{'Flagged' if result['flagged'] else 'Not flagged'} at a cut-off of "
        f"{result['threshold']:.4f}."
    )
    st.caption(
        "Scored just now and recorded." if is_fresh
        else "The last score recorded for this customer."
    )

    st.markdown("**Why**")
    for reason in result["reasons"]:
        raises = reason.get("raises_risk", reason.get("effect", 0) > 0)
        arrow = "▲" if raises else "▼"
        colour = "#cc3b2f" if raises else "#2e9e5b"
        st.markdown(
            f"<span style='color:{colour}'>{arrow}</span> {reason['description']}",
            unsafe_allow_html=True,
        )
    st.caption("▲ raises the risk, ▼ lowers it, strongest first.")


def _decision(detail: dict, officer_email: str) -> None:
    st.markdown("**Decision**")

    try:
        trail = api.decisions(detail["customer_id"])
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    if trail:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When": entry["decided_at"][:16].replace("T", " "),
                        "Action": ACTIONS.get(entry["action"], entry["action"]),
                        "Justification": entry["justification"],
                    }
                    for entry in trail
                ]
            ),
            hide_index=True, width="stretch",
        )

    with st.form(f"decide_{detail['customer_id']}"):
        action = st.radio(
            "Action", list(ACTIONS), horizontal=True,
            format_func=lambda key: ACTIONS[key],
        )
        justification = st.text_area(
            "Justification",
            placeholder="Why this action, in your own words.",
            help="Required on every decision, not only overrides, so the audit "
                 "trail has no holes.",
        )
        submitted = st.form_submit_button("Record decision", type="primary")

    if not submitted:
        return
    if not justification.strip():
        st.error("A justification is required before a decision can be recorded.")
        return

    try:
        api.decide(detail["customer_id"], officer_email, action, justification)
    except api.ServiceUnavailable as exception:
        st.error(str(exception))
        return

    api.decisions.clear()
    st.success(f"**{ACTIONS[action]}** recorded for {detail['customer_id']}.")
    st.rerun()


# --- the triage tab -------------------------------------------------------


def _queue() -> None:
    st.caption(
        "Everyone scored in this month's batch, riskiest first. Bands only: the "
        "precise risk and the reasons come from running the prediction."
    )

    left, middle, right = st.columns([2, 2, 3])
    band = left.selectbox("Risk band", ["All", *BANDS])
    flagged_only = middle.checkbox("Needs attention only", value=True)
    arrears = right.slider("Minimum months in arrears", 0, 6, 0)

    try:
        page = api.queue(
            band=None if band == "All" else band,
            flagged=True if flagged_only else None,
            minimum_months_in_arrears=arrears or None,
            limit=PAGE_SIZE,
        )
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    everything = page["summary_all"]
    for column, (label, value, explanation) in zip(
        st.columns(4),
        [
            ("In queue", f"{everything['in_queue']:,}", "Customers scored this month."),
            ("Needs attention", f"{everything['flagged']:,}",
             f"At or above the cut-off of {page['threshold']:.4f}."),
            ("High risk", f"{everything['bands']['High']:,}", "In the top band."),
            ("Low risk", f"{everything['bands']['Low']:,}", "In the safest band."),
        ],
    ):
        column.metric(label, value, help=explanation)

    st.caption(f"Showing {page['returned']} of {page['total']:,} matching customers.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Customer": row["customer_id"],
                    "Risk": f"{BAND_ICONS[row['risk_band']]} {row['risk_band']}",
                    "Credit limit": f"{row['credit_limit']:,.0f}",
                    "Balance": f"{row['current_balance']:,.0f}",
                    "In arrears": f"{row['months_in_arrears']} of 6",
                    "Needs attention": "Yes" if row["flagged"] else "-",
                }
                for row in page["rows"]
            ]
        ),
        hide_index=True, width="stretch",
    )
    st.caption("Copy an account number into **Find a customer** to open it.")
