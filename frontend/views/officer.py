"""Loan officer view: search, review requests, and a worklist Loana helps fill.

CrediTrait is a behavioural scorecard for existing customers. It raises an early
warning; it does not approve or deny anyone. A customer who thinks the rating is
wrong can ask for a human review, and the officer decides whether it should stand.

Two states on one page, no tabs. With nobody selected it is a landing page: a large
search box, the review requests (the action items), and an empty worklist beside
Loana's triage panel. The officer asks Loana for a cohort, ticks customers and runs
"predict selected"; no score is shown until someone runs one. Once the officer picks
a customer, the page becomes the customer's record at full width with Loana AI as
a floating chat about that customer (ui/loana.py).

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

import pandas as pd
import streamlit as st

import api
import review
from components import BAND_ICONS
from ui import loana, thread
from ui.components import card_head, quote, reason_line, show
from views.cardholder import ACTIONABLE, history_card

ACTIONS = review.ACTIONS
# Customers named in the landing alert; the full cards below list the rest.
INBOX_SHORTCUTS = 4
# Badges past these read as "99+" / "9+", so a backlog cannot break the layout.
BADGE_CAP = 99
UNREAD_CAP = 9
# Request cards drawn on the landing, longest waiting first; each one costs a lookup.
REQUEST_CARDS = 10
# The backend caps "predict selected" at the same number.
MAX_PREDICT = 10

# No model is certain, so the extremes are shown as bounds.
CERTAINTY_CAP = 0.99

CUSTOMER_KEY = "officer_customer_id"
TRIAGE_CHAT_KEY = "officer_triage_chat"
TRIAGE_INPUT_KEY = "officer_triage_ask"
TRIAGE_PANEL_KEY = "officer_triage"
COHORT_KEY = "officer_cohort"
COHORT_FILTER_KEY = "officer_cohort_filter"
# The table's ticks live in the browser, so clearing them needs a new widget key.
COHORT_EDITOR_KEY = "officer_cohort_editor"
EDITOR_VERSION_KEY = "officer_cohort_editor_version"
COHORT_ERROR_KEY = "officer_cohort_error"
JUST_SCORED_KEY = "officer_just_scored"
LOOKUP_KEY = "officer_lookup"
NOTICE_KEY = "officer_notice"

# Widget state Streamlit drops when its widget is not drawn (another customer,
# another view). Re-assigned each run so a half-typed note survives the detour.
KEPT_PREFIXES = ("officer_note_", "officer_action_")

PRESETS = (
    "Why is this customer flagged?",
    "What changed recently?",
    "What would lower the rating?",
)
TRIAGE_HELP = (
    "Tell me which customers to pull into the worklist: by months behind, whether "
    "it is getting worse, utilisation, missed payments, limit, or not scored yet. "
    "I don't score anyone; you run the prediction."
)
TRIAGE_CHAT_HEIGHT = 230
EMPTY_WORKLIST = (
    "No customers pulled in yet. Ask Loana for a group, such as *customers three "
    "months behind and getting worse*, or pick a quick cohort."
)

# Preset filters, the same shape Loana fills. Unscored only, so working through a
# cohort empties it.
QUICK_COHORTS = {
    "Getting worse, 2+ months behind": {
        "min_months_in_arrears": 2, "trajectory": "deteriorating", "not_scored": True,
    },
    "No payment in 3+ months": {
        "min_months_with_no_payment": 3, "sort_by": "months_with_no_payment",
        "not_scored": True,
    },
    "Using 90%+ of their limit": {
        "min_utilisation": 0.9, "sort_by": "utilisation_now", "not_scored": True,
    },
}

WORKLIST_COLUMNS = (
    "Select", "Messages", "Account", "Customer", "In arrears", "Trend", "Used",
    "No payment", "Balance", "Risk",
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
        _landing(officer_email)
        return

    header()
    _record_page(customer_id, officer_email, token)
    loana.dock(lambda: _assistant(customer_id))


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


def back_button() -> None:
    """The way out of a customer page; app.py pins it into the header bar."""
    if st.session_state.get(CUSTOMER_KEY):
        st.button("Back to worklist", key="officer_close", on_click=_close_profile)


def _close_profile() -> None:
    _select(None)
    st.session_state[LOOKUP_KEY] = ""


# --- review requests ------------------------------------------------------


def _all_requests() -> list[dict]:
    """Every open cardholder message; empty when the service is down, so the page still draws."""
    try:
        return api.review_requests()
    except api.ServiceUnavailable:
        return []


def _inbox_alert(requests: list[dict]) -> None:
    """Top of the landing page: who is waiting for a reply, one click from their thread."""
    waiting = review.conversations(requests)
    if not waiting:
        show('<div class="inbox-alert inbox-alert--clear">✓ No customers are waiting '
             "for a reply.</div>")
        return

    unread = sum(item["unread"] for item in waiting)
    people = "customer is" if len(waiting) == 1 else "customers are"
    show(
        f'<div class="inbox-alert"><span class="inbox-badge">'
        f"{review.capped(len(waiting), BADGE_CAP)}</span>"
        f"<div><b>🔔 {len(waiting):,} {people} waiting for a reply</b> · {unread:,} unread "
        f"message{'' if unread == 1 else 's'}. Recording a decision answers them and "
        "closes the conversation.</div></div>"
    )
    for column, item in zip(st.columns(INBOX_SHORTCUTS), waiting[:INBOX_SHORTCUTS]):
        name = item.get("full_name") or f"Customer {item['account_number']}"
        column.button(
            f"🔔 {name} ({review.capped(item['unread'], UNREAD_CAP)})",
            key=f"inbox_{item['customer_id']}",
            on_click=_select, args=(item["customer_id"],), width="stretch",
        )


def _requests(requests: list[dict]) -> None:
    waiting = review.conversations(requests)
    st.subheader("Customers asking for a human review")
    st.caption(
        f"{len(waiting)} waiting · {review.pending_count(requests)} unread messages. "
        "A review is when a person decides whether the model's rating should stand."
    )
    if not waiting:
        st.info("No customers are waiting for a review.")
        return

    now = datetime.now(timezone.utc)
    for item in waiting[:REQUEST_CARDS]:
        _request_card(item, _band(item["customer_id"]), now)
    hidden = len(waiting) - REQUEST_CARDS
    if hidden > 0:
        st.caption(
            f"Showing the {REQUEST_CARDS} longest waiting. {hidden:,} more are waiting; "
            "each one is flagged 🔔 in the worklist and opens from search."
        )


def _band(customer_id: str) -> str | None:
    """The band from the latest prediction an officer ran, or None if nobody has."""
    try:
        latest = api.customer_detail(customer_id)["latest_prediction"]
    except api.ServiceUnavailable:
        return None
    return latest["risk_band"] if latest else None


def _request_card(item: dict, band: str | None, now: datetime) -> None:
    card = review.card_data({**item, "reason": item["latest"]}, band, now)
    unread = item["unread"]
    with st.container(border=True):
        show(card_head(
            card["name"],
            f"Account {card['account']} · waiting {card['waiting']} · "
            f"{unread} unread message{'' if unread == 1 else 's'}",
            card["band"],
            lift=True,
        ))
        show(quote(card["note"]))
        if card["is_truncated"]:
            with st.expander("Read the full note"):
                show(quote(card["full_note"]))
        st.button(
            "Open conversation", key=f"review_{item['request_id']}",
            on_click=_select, args=(item["customer_id"],),
        )


# --- the landing page -----------------------------------------------------


def _landing(officer_email: str) -> None:
    """Nobody is open: search, the action items, then the worklist beside Loana."""
    # Search and worklist share the wide column and Loana sticks beside it, so the
    # table headers and her input are on screen at load. Review requests follow.
    requests = _all_requests()
    _inbox_alert(requests)

    worklist, assistant = st.columns([7, 3], gap="large")
    with worklist:
        typed = st.text_input(
            "Search", key=LOOKUP_KEY, label_visibility="collapsed",
            placeholder="Find a customer by ID or name...",
        )
        if typed.strip():
            _results(typed.strip())
        _worklist(officer_email, review.unread_by_customer(requests))
    with assistant, st.container(border=True, key=TRIAGE_PANEL_KEY):
        _triage()

    st.divider()
    _requests(requests)


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
    """This customer's open conversation. Without one the page is a plain profile."""
    for item in review.conversations(_all_requests()):
        if item["customer_id"] == customer_id:
            return item
    return None


def _label(match: dict) -> str:
    return f"{match['full_name']} · {match['account_number']}"


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
        _unread_alert(request)
        _evidence(detail, request)
        _plan(detail, token)
    _figures(detail)
    st.divider()
    _history(detail)
    st.divider()
    _prediction(detail, officer_email)
    st.divider()
    _activity(detail, officer_email, request is not None)


# --- the review itself ----------------------------------------------------


def _unread_alert(request: dict) -> None:
    """The customer is waiting: say so above everything else on their record."""
    unread = request["unread"]
    waited = review.waiting_text(request["requested_at"], datetime.now(timezone.utc))
    show(
        f'<div class="inbox-alert"><span class="inbox-badge">'
        f"{review.capped(unread, UNREAD_CAP)}</span>"
        f"<div><b>🔔 {unread} unanswered message{'' if unread == 1 else 's'} from this "
        f"customer</b> · waiting {waited}. Reply by recording a decision in the "
        "activity history below.</div></div>"
    )


def _evidence(detail: dict, request: dict) -> None:
    saw, says = st.columns(2, gap="large")
    with saw:
        st.markdown("**What the model saw**")
        st.caption("The last six months of this account.")
        for sentence in review.model_saw(detail):
            st.markdown(f"- {sentence}")
    with says:
        st.markdown("**What the customer says**")
        show(quote(request["latest"].strip()))


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

    name = detail.get("full_name")
    st.subheader(name or f"Customer {detail['account_number']}")

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
    history_card(detail["history"], "Last six months")

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
        f"{'Flagged for attention' if result['flagged'] else 'Not flagged for attention'}."
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


def _activity(detail: dict, officer_email: str, from_request: bool) -> None:
    """The account's shared thread, with the decision composer at its foot.

    The cardholder reads the same thread, so the note is written to them too.
    """
    customer_id = detail["customer_id"]
    st.markdown("**Activity & communication history**")
    st.caption(
        "Messages from the customer and every decision, oldest first. "
        "The customer sees this same history."
    )

    try:
        entries = api.thread(customer_id)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return
    thread.draw(entries, "officer")

    with st.container(border=True, key="thread_composer"):
        st.markdown("**Reply with a decision**" if from_request else "**Record a decision**")
        # Widgets rather than a form: a form's unsent text never reaches session
        # state, so it could not survive a trip to another customer.
        st.radio(
            "Action", list(ACTIONS), horizontal=True, key=f"officer_action_{customer_id}",
            format_func=lambda key: ACTIONS[key],
        )
        st.text_area(
            "Note to the customer",
            key=f"officer_note_{customer_id}",
            placeholder="Why this action, in words the customer will understand.",
            help="Required on every decision, so the audit trail has no holes. "
                 "The customer sees it in their message history.",
        )
        st.button(
            "Record decision", type="primary", key=f"officer_record_{customer_id}",
            on_click=_record,
            args=(
                customer_id, officer_email, detail.get("full_name"),
                detail["account_number"], from_request,
            ),
        )
        if from_request:
            st.caption(
                "Recording a decision answers the open messages and closes the "
                "conversation until the customer writes again."
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
        notify(False, "A note is required before a decision can be recorded.")
        return

    try:
        api.decide(customer_id, officer_email, action, justification)
    except api.ServiceUnavailable as exception:
        notify(False, str(exception))
        return

    st.session_state[note_key] = ""
    who = f"**{name}** (account {account_number})" if name else f"account **{account_number}**"
    outcome = (
        "Decision recorded and conversation closed" if from_request else "Decision recorded"
    )
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


# --- the worklist and Loana's triage ---------------------------------------


def _show_cohort(page: dict, criteria: dict) -> None:
    """Make this cohort the worklist, forgetting the previous one's ticks."""
    st.session_state[COHORT_KEY] = page
    st.session_state[COHORT_FILTER_KEY] = criteria
    st.session_state.pop(JUST_SCORED_KEY, None)
    _clear_ticks()


def _clear_ticks() -> None:
    st.session_state[EDITOR_VERSION_KEY] = st.session_state.get(EDITOR_VERSION_KEY, 0) + 1


def _quick_cohort(criteria: dict) -> None:
    """Button callback: pull a preset cohort without asking Loana."""
    try:
        _show_cohort(api.cohort(criteria), criteria)
    except api.ServiceUnavailable as exception:
        st.session_state[COHORT_ERROR_KEY] = str(exception)


def _triage() -> None:
    """Loana on the landing page: the officer describes a group, she pulls it in.

    She only fills a filter; the backend checks it and writes her reply from it,
    so nothing she says can name a customer or a risk the data does not back.
    """
    loana.header("Triage assistant")
    messages = st.session_state.setdefault(TRIAGE_CHAT_KEY, [])

    chat = st.container(height=TRIAGE_CHAT_HEIGHT)
    with chat:
        st.caption(TRIAGE_HELP)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("note"):
                    st.caption(message["note"])

    st.caption("Quick cohorts")
    for number, (label, criteria) in enumerate(QUICK_COHORTS.items()):
        st.button(
            label, key=f"officer_quick_{number}", width="stretch",
            on_click=_quick_cohort, args=(criteria,),
        )

    asked = st.chat_input("Describe who to look at", key=TRIAGE_INPUT_KEY)
    if not asked:
        return

    sent = [
        *({"role": m["role"], "content": m["content"]} for m in messages),
        {"role": "user", "content": asked},
    ]
    with chat:
        with st.chat_message("user"):
            st.markdown(asked)
        with st.spinner("Finding customers..."):
            try:
                turn = api.triage(sent)
            except api.ServiceUnavailable:
                st.info("Assistant not connected yet.")
                return

    reply = turn["reply"]
    cohort = turn.get("cohort")
    if cohort:
        _show_cohort(cohort, turn["filter"])
        reply += (
            f" {cohort['total']:,} match; the worklist shows the first "
            f"{len(cohort['rows'])}."
        )
    messages.append({"role": "user", "content": asked})
    messages.append({"role": "assistant", "content": reply, "note": turn.get("note", "")})
    st.rerun()


def _worklist(officer_email: str, unread: dict[str, int]) -> None:
    """The cohort Loana or a quick cohort pulled in, to tick, predict and open."""
    st.subheader("Worklist")
    if st.session_state.pop(COHORT_ERROR_KEY, None):
        st.warning("Could not load that cohort. Is the API running?")

    page = st.session_state.get(COHORT_KEY)
    if not page:
        st.info(EMPTY_WORKLIST)
        return

    _just_scored()

    st.caption(f"{page['description']} Showing {len(page['rows'])} of {page['total']:,}.")
    rows = page["rows"]
    if not rows:
        st.info("Nobody matches. Try a wider group.")
        return

    edited = st.data_editor(
        pd.DataFrame(
            [_worklist_row(row, unread.get(row["customer_id"], 0)) for row in rows]
        ),
        key=f"{COHORT_EDITOR_KEY}_{st.session_state.get(EDITOR_VERSION_KEY, 0)}",
        hide_index=True, width="stretch",
        disabled=[column for column in WORKLIST_COLUMNS if column != "Select"],
        column_config={"Select": st.column_config.CheckboxColumn(width="small")},
    )
    chosen = [row for row, ticked in zip(rows, edited["Select"]) if ticked]

    predict, view = st.columns(2)
    too_many = len(chosen) > MAX_PREDICT
    if predict.button(
        f"Predict selected ({len(chosen)})", type="primary", width="stretch",
        disabled=not chosen or too_many, key="officer_predict_selected",
    ):
        _predict_selected(chosen, officer_email)
    if too_many:
        predict.caption(f"Up to {MAX_PREDICT} at a time.")

    only = chosen[0] if len(chosen) == 1 else None
    view.button(
        f"View analysis: {only['full_name']}" if only else "View analysis",
        width="stretch", disabled=only is None, key="officer_view_analysis",
        on_click=_select, args=(only["customer_id"] if only else None,),
    )
    if len(chosen) > 1:
        view.caption("Tick one customer to open their analysis.")


def _just_scored() -> None:
    """The customers the last "predict selected" scored, each one click from its analysis."""
    scored = st.session_state.get(JUST_SCORED_KEY)
    if not scored:
        return
    flagged = sum(1 for item in scored if item["flagged"])
    with st.container(border=True):
        st.markdown(
            f"**Scored {len(scored)}** · {flagged} "
            f"{'needs' if flagged == 1 else 'need'} attention. Open one to see why."
        )
        for item in scored:
            st.button(
                f"{BAND_ICONS[item['risk_band']]} {item['risk_band']} · {item['label']}",
                key=f"officer_scored_{item['customer_id']}", width="stretch",
                on_click=_select, args=(item["customer_id"],),
            )


def _worklist_row(row: dict, unread: int = 0) -> dict:
    band = row["last_risk_band"]
    utilisation = row["utilisation_now"]
    return {
        "Select": False,
        # Customers waiting on a reply stand out in whatever cohort they land in.
        "Messages": f"🔔 {review.capped(unread, UNREAD_CAP)} new" if unread else "",
        "Account": row["account_number"],
        "Customer": row["full_name"],
        "In arrears": f"{row['months_in_arrears']} of 6",
        "Trend": (row["trajectory"] or "-").capitalize(),
        "Used": "-" if utilisation is None else f"{utilisation:.0%}",
        "No payment": f"{row['months_with_no_payment']} of 6",
        "Balance": f"{row['current_balance']:,.0f}",
        "Risk": f"{BAND_ICONS[band]} {band}" if band else "Not scored",
    }


def _predict_selected(chosen: list[dict], officer_email: str) -> None:
    """Score the ticked customers and write each band into the worklist."""
    with st.spinner(f"Predicting for {len(chosen)} customers..."):
        try:
            scores = api.predict_selected(
                [row["customer_id"] for row in chosen], scored_by=officer_email
            )
        except api.ServiceUnavailable as exception:
            api.guard(exception)
            return

    by_id = {row["customer_id"]: score for row, score in zip(chosen, scores)}
    for row in st.session_state[COHORT_KEY]["rows"]:
        if row["customer_id"] in by_id:
            row["last_risk_band"] = by_id[row["customer_id"]]["risk_band"]
    _clear_ticks()
    st.session_state[JUST_SCORED_KEY] = [
        {
            "customer_id": row["customer_id"],
            "label": _label(row),
            "risk_band": score["risk_band"],
            "flagged": score["flagged"],
        }
        for row, score in zip(chosen, scores)
    ]
    st.rerun()
