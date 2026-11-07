"""Loan officer view: search first, then review requests and this month's triage.

CrediTrait is a behavioural scorecard for existing customers. It raises an early
warning; it does not approve or deny anyone. A customer who thinks the rating is
wrong can ask for a human review, and the officer decides whether it should stand.

Two states on one page, no tabs. With nobody selected it is a landing page: a large
search box, the review requests and this month's triage, and no assistant. Once the
officer picks a customer, from a search result or a request card, the page becomes
the customer's record at 70% with Loana AI at 30% (ui/loana.py).

Nothing opens by itself: the selected customer (`officer_customer_id`) is only ever
set by a click, and is deliberately not mirrored in or read from the URL, so a
stale link cannot open a customer the officer never chose.

Search is a way in, not a leaderboard: the actions on offer are account
management, which starts from a customer rather than a ranking. Demographics never
appear. They are dropped before scoring (conventions §10) and putting them beside
a score would invite the discrimination the audit rules out.
"""
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import api
import review
import theme
from components import BAND_ICONS
from ui import loana
from ui.components import card_head, quote, reason_line, show
from views.cardholder import ACTIONABLE

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

STATUS_COLOURS = {"arrears": theme.HIGH, "revolving": theme.MEDIUM, "clear": theme.LOW}

CUSTOMER_KEY = "officer_customer_id"
LOOKUP_KEY = "officer_lookup"
NOTICE_KEY = "officer_notice"

# Widget state Streamlit drops when its widget is not drawn (another customer,
# another view). Re-assigned each run so a half-typed note survives the detour.
KEPT_PREFIXES = ("officer_note_", "officer_action_")

HERO = Path(__file__).resolve().parent.parent / "assets" / "review_hero.png"

PRESETS = (
    "Why is this customer flagged?",
    "What changed recently?",
    "What would lower the rating?",
)
ASSISTANT_HELP = (
    "Answers questions about this customer's account and the model's reasons. "
    "Lending policy questions are not supported yet."
)


def render(
    officer_email: str, token: str, header: Callable[[], None] = loana.no_header
) -> None:
    """`header` draws the page bar: full width on the landing, in the main column after."""
    _keep_widget_state()

    customer_id = st.session_state.get(CUSTOMER_KEY)
    if not customer_id:
        header()
        _landing()
        return

    main, assistant = loana.columns()
    with main:
        header()
        _record_page(customer_id, officer_email, token)
    with assistant, loana.panel():
        _assistant(customer_id)


def _keep_widget_state() -> None:
    for key in list(st.session_state):
        if key == LOOKUP_KEY or key.startswith(KEPT_PREFIXES):
            st.session_state[key] = st.session_state[key]


# --- the selected customer ------------------------------------------------


def _select(customer_id: str | None) -> None:
    """Open this customer, or None to go back to the landing page.

    Used as a button callback, which is the only way a customer is ever opened.
    """
    if st.session_state.get(CUSTOMER_KEY) == customer_id:
        return
    st.session_state[CUSTOMER_KEY] = customer_id
    st.session_state.pop(NOTICE_KEY, None)


def _close_profile() -> None:
    _select(None)
    st.session_state[LOOKUP_KEY] = ""


# --- review requests ------------------------------------------------------


def _requests() -> None:
    try:
        requests = api.review_requests()
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    waiting = review.open_requests(requests)
    left, right = st.columns([2, 3], gap="large")

    with left:
        st.image(str(HERO), width="stretch")
        st.subheader("Customers asking for a human review")
        st.write(
            "A review is when a person looks at the account and decides whether "
            "the model's rating of the customer should stand."
        )

    with right:
        st.markdown(f"**{review.pending_count(requests)} pending** · {len(waiting)} open")
        if not waiting:
            st.info("No customers are waiting for a review.")
            return

        now = datetime.now(timezone.utc)
        for item in waiting:
            _request_card(item, _band(item["customer_id"]), now)


def _band(customer_id: str) -> str | None:
    """The model's current band; a card without one is better than no card."""
    try:
        return api.customer_detail(customer_id)["risk_band"]
    except api.ServiceUnavailable:
        return None


def _request_card(item: dict, band: str | None, now: datetime) -> None:
    card = review.card_data(item, band, now)
    with st.container(border=True):
        show(card_head(
            card["name"],
            f"Account {card['account']} · waiting {card['waiting']} · {card['status']}",
            card["band"],
        ))
        show(quote(card["note"]))
        if card["is_truncated"]:
            with st.expander("Read the full note"):
                show(quote(card["full_note"]))
        st.button(
            "Review", key=f"review_{item['request_id']}",
            on_click=_select, args=(item["customer_id"],),
        )


# --- the landing page -----------------------------------------------------


def _landing() -> None:
    """Nobody is open: search, then the review requests and this month's triage."""
    st.subheader("Find a customer")
    st.caption(
        "Look up the customer in front of you and check whether their next "
        "payment is at risk."
    )
    typed = st.text_input(
        "Search", key=LOOKUP_KEY, label_visibility="collapsed",
        placeholder="Search by Customer ID or Name...",
    )
    if typed.strip():
        _results(typed.strip())
    else:
        _suggestions()

    st.divider()
    _requests()
    st.divider()
    st.subheader("This month's triage")
    _queue()


def _results(typed: str) -> None:
    """Matches as buttons. Nothing opens until one is clicked, even a lone match."""
    try:
        matches = api.search_customers(typed)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    if not matches:
        st.warning(f"No customer matches {typed!r}.")
        return

    st.caption(f"{len(matches)} matching. Pick one to open the account.")
    for match in matches:
        st.button(
            _label(match), key=f"officer_pick_{match['customer_id']}",
            on_click=_select, args=(match["customer_id"],), width="stretch",
        )


def _open_request_for(customer_id: str) -> dict | None:
    """This customer's open review request. Without one the page is a plain profile."""
    try:
        return review.request_for(api.review_requests(), customer_id)
    except api.ServiceUnavailable:
        return None


def _label(match: dict) -> str:
    return f"{match['full_name']} · {match['account_number']}"


def _suggestions() -> None:
    """Accounts to open without typing, so the screen is never a dead end."""
    try:
        known = api.users(role="cardholder")
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return
    registered = [user for user in known if user["customer_id"]]
    if registered:
        st.caption("Try: " + ", ".join(
            f"**{user['display_name']}** (`{user['account_number']}`)"
            for user in registered
        ))


# --- the customer's record ------------------------------------------------


def _record_page(customer_id: str, officer_email: str, token: str) -> None:
    try:
        detail = api.customer_detail(customer_id)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    request = _open_request_for(customer_id)
    _heading(detail)
    if request:
        _review_banner(request)
        _evidence(detail, request)
        _plan(detail, token)
    _figures(detail)
    st.divider()
    _history(detail)
    st.divider()
    _prediction(detail, officer_email)
    st.divider()
    _decision(detail, officer_email, request is not None)


# --- the review itself ----------------------------------------------------


def _review_banner(request: dict) -> None:
    """The request as the customer made it, and what they told the assistant first."""
    now = datetime.now(timezone.utc)
    with st.container(border=True):
        show(card_head(
            "Review request",
            f"Received {review.format_time(request['requested_at'])} · "
            f"waiting {review.waiting_text(request['requested_at'], now)}",
        ))
        show(quote(request["reason"].strip()))

        try:
            turns = api.chat_before_request(
                request["customer_id"], request["requested_at"]
            )
        except api.ServiceUnavailable:
            st.caption("The earlier assistant conversation could not be loaded.")
            return
        if not turns:
            st.caption("No assistant conversation before this request.")
            return
        with st.expander(f"Assistant conversation before the request ({len(turns)} turns)"):
            for turn in turns:
                who = "Customer" if turn["role"] == "user" else "Assistant"
                st.caption(f"{who} · {review.format_time(turn['at'])}")
                st.write(turn["content"])


def _evidence(detail: dict, request: dict) -> None:
    saw, says = st.columns(2, gap="large")
    with saw:
        st.markdown("**What the model saw**")
        st.caption("The last six months of this account.")
        for sentence in review.model_saw(detail):
            st.markdown(f"- {sentence}")
    with says:
        st.markdown("**What the customer says**")
        show(quote(request["reason"].strip()))


def _plan(detail: dict, token: str) -> None:
    """What would move the rating, in the words the customer is shown."""
    st.markdown("**What would change the rating**")

    try:
        plan = api.recourse(detail["customer_id"], token)
    except api.ServiceUnavailable:
        # The rest of the review stands without it.
        st.caption("The search for what would change the rating could not be reached.")
        return

    if plan["steps"]:
        st.markdown(f"To move to {plan['target_band'].lower()} risk:")
        for step in plan["steps"]:
            st.markdown(f"- {step['description']}")
        if plan["reachable"]:
            st.caption(plan["note"])
        else:
            st.caption(
                f"{plan['note']} Each one still helps, and support can look at "
                "options this account cannot reach on its own."
            )
    else:
        st.caption(plan["note"])

    unchangeable = review.unchangeable_reason(detail["reasons"], set(ACTIONABLE))
    if unchangeable:
        st.info(
            f"The model's strongest reason for the risk, \"{unchangeable.lower()}\", "
            "is not something the customer can change directly."
        )


# --- the customer page ----------------------------------------------------


def _heading(detail: dict) -> None:
    segment = detail.get("segment") or {}

    title, close = st.columns([4, 2])
    name = detail.get("full_name")
    title.subheader(name or f"Customer {detail['account_number']}")
    close.button("Back to search", key="officer_close", on_click=_close_profile, width="stretch")

    st.caption(f"Account {detail['account_number']}")
    if segment:
        st.caption(f"Behavioural segment: **{segment['name']}**")


def _figures(detail: dict) -> None:
    highlights = detail["highlights"]
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
    st.altair_chart(bars, width="stretch", theme=None)

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
        show(reason_line(reason["description"], raises))
    st.caption("▲ raises the risk, ▼ lowers it, strongest first.")


def _decision(detail: dict, officer_email: str, from_request: bool) -> None:
    customer_id = detail["customer_id"]
    st.markdown(
        "**Record the outcome of this review**" if from_request else "**Decision**"
    )

    try:
        trail = api.decisions(customer_id)
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

    # Widgets rather than a form: a form's unsent text never reaches session state,
    # so it could not survive a trip to another customer.
    st.radio(
        "Action", list(ACTIONS), horizontal=True, key=f"officer_action_{customer_id}",
        format_func=lambda key: ACTIONS[key],
    )
    st.text_area(
        "Justification",
        key=f"officer_note_{customer_id}",
        placeholder="Why this action, in your own words.",
        help="Required on every decision, not only overrides, so the audit "
             "trail has no holes.",
    )
    st.button(
        "Record decision", type="primary", key=f"officer_record_{customer_id}",
        on_click=_record,
        args=(
            customer_id, officer_email, detail.get("full_name"),
            detail["account_number"], from_request,
        ),
    )

    notice = st.session_state.get(NOTICE_KEY)
    if notice and notice["customer_id"] == customer_id:
        (st.success if notice["ok"] else st.error)(notice["text"])


def _record(
    customer_id: str, officer_email: str, name: str | None,
    account_number: str, from_request: bool,
) -> None:
    """Save the decision. A callback, so it can clear the note that was just sent."""
    action = st.session_state[f"officer_action_{customer_id}"]
    note_key = f"officer_note_{customer_id}"
    justification = st.session_state.get(note_key, "")

    def notify(ok: bool, text: str) -> None:
        st.session_state[NOTICE_KEY] = {
            "customer_id": customer_id, "ok": ok, "text": text,
        }

    if not justification.strip():
        notify(False, "A justification is required before a decision can be recorded.")
        return

    try:
        api.decide(customer_id, officer_email, action, justification)
    except api.ServiceUnavailable as exception:
        notify(False, str(exception))
        return

    api.decisions.clear()
    st.session_state[note_key] = ""
    who = f"**{name}** (account {account_number})" if name else f"account **{account_number}**"
    outcome = "Review outcome recorded" if from_request else "Decision recorded"
    notify(True, f"{outcome}: **{ACTIONS[action]}** for {who}.")


# --- the customer assistant -----------------------------------------------


def _assistant(customer_id: str) -> None:
    """Loana AI for the officer, scoped to the open customer.

    History is keyed by customer, so one customer's facts never reach another's
    conversation. Every turn goes through `api.officer_chat`, never the cardholder
    endpoint, which hides the probability on purpose.
    """
    messages = st.session_state.setdefault(f"officer_chat_{customer_id}", [])

    # Messages scroll in their own box; presets and the input stay under it.
    chat = loana.chat_area()
    with chat:
        st.caption(ASSISTANT_HELP)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("note"):
                    st.caption(message["note"])

    asked = None
    for number, question in enumerate(PRESETS):
        if st.button(
            question, key=f"officer_preset_{customer_id}_{number}", width="stretch"
        ):
            asked = question

    asked = st.chat_input("Ask about this customer", key="officer_ask") or asked
    if not asked:
        return

    sent = [
        *({"role": m["role"], "content": m["content"]} for m in messages),
        {"role": "user", "content": asked},
    ]
    with chat:
        with st.chat_message("user"):
            st.markdown(asked)
        with st.spinner("Reading the account..."):
            try:
                turn = api.officer_chat(customer_id, sent)
            except api.ServiceUnavailable:
                st.info("Assistant not connected yet.")
                return

    messages.append({"role": "user", "content": asked})
    messages.append(
        {"role": "assistant", "content": turn["reply"], "note": turn.get("note", "")}
    )
    st.rerun()


# --- the triage view ------------------------------------------------------


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
                    "Account": row["account_number"],
                    "Customer": row["full_name"],
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
    st.caption("Copy an account number into the search box above to open it.")
