"""Loan officer view: search, review requests, and a worklist Loana helps fill.

CrediTrait is a behavioural scorecard for existing customers. It raises an early
warning; it does not approve or deny anyone. A customer who thinks the rating is
wrong can ask for a human review, and the officer decides whether it should stand.

Two states on one page, no tabs. With nobody selected it is a landing page led by
Loana: a briefing written from the worklist counts, one command bar that either
finds a customer or asks Loana for a cohort, the cohort's filter as removable
chips, and the worklist. The officer ticks customers and runs "predict selected";
no score is shown until someone runs one. Once the officer picks
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
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import api
import review
from money import markdown_safe, money
from ui import loana, nav, thread
from ui.profile import initials
from ui.components import card_head, quote, show, stylesheet
from views import prediction
from views.cardholder import ACTIONABLE, history_card

ACTIONS = review.ACTIONS
# Badges past these read as "99+" / "9+", so a backlog cannot break the layout.
BADGE_CAP = 99
UNREAD_CAP = 9
# Request cards drawn on the landing, longest waiting first; each one costs a lookup.
REQUEST_CARDS = 10
# The backend caps "predict selected" at the same number.
MAX_PREDICT = 10

CUSTOMER_KEY = "officer_customer_id"
CONVERSATION_KEY = "officer_conversation_id"
# Only the triage turns, which are what Loana is sent; searches are not.
TRIAGE_CHAT_KEY = "officer_triage_chat"
COMMAND_KEY = "officer_command"
BRIEFING_KEY = "officer_briefing"
# The briefing types itself out once per session, then stays still.
BRIEFED_KEY = "officer_briefed"
EXCHANGE_KEY = "officer_exchange"
SEARCH_KEY = "officer_search"
COHORT_KEY = "officer_cohort"
COHORT_FILTER_KEY = "officer_cohort_filter"
# Ticks and chips are widgets, so resetting them needs fresh keys.
EDITOR_VERSION_KEY = "officer_cohort_editor_version"
TICK_PREFIX = "officer_tick_"
CHIPS_KEY = "officer_chips"
COHORT_ERROR_KEY = "officer_cohort_error"
JUST_SCORED_KEY = "officer_just_scored"
NOTICE_KEY = "officer_notice"

# Widget state Streamlit drops when its widget is not drawn (another customer,
# another view). Re-assigned each run so a half-typed note survives the detour.
KEPT_PREFIXES = ("officer_note_", TICK_PREFIX)

PRESETS = (
    "Why is this customer flagged?",
    "What changed recently?",
    "What would lower the rating?",
)
COMMAND_PLACEHOLDER = (
    "Ask LOANA for a group, or find a customer by name or account..."
)
EMPTY_WORKLIST = (
    "No customers pulled in yet. Pick one of LOANA's suggestions above, or ask for "
    "a group, such as *customers three months behind and getting worse*."
)
# api.search_customers' default page size.
SEARCH_LIMIT = 25
# Seconds per word when Loana's text types itself out.
TYPING_DELAY = 0.025

# Words that mean the officer is describing a group, not naming a customer.
COHORT_WORDS = {
    "behind", "arrears", "late", "worse", "improving", "steady", "deteriorating",
    "month", "months", "payment", "payments", "paid", "limit", "utilisation",
    "utilization", "using", "used", "scored", "unscored", "customers", "who",
    "with", "balance", "missed", "over", "under", "all", "top",
}

# The filter fields shown as chips, in reading order; sort and limit are not.
CHIP_FIELDS = (
    "min_months_in_arrears", "trajectory", "min_months_with_no_payment",
    "min_utilisation", "min_credit_limit", "max_credit_limit", "not_scored", "scored_band",
)

# Relative widths of the worklist's columns: tick, five facts, the dashboard link.
WORKLIST_HEADINGS = (
    "", "Account", "Customer", "In arrears", "Trend", "Used", "No payment", "Balance", "",
)
WORKLIST_WIDTHS = (0.45, 0.9, 2.1, 0.9, 1.1, 0.7, 0.95, 1.0, 2.0)

ASSISTANT_HELP = (
    "Answers questions about this customer's account and the model's reasons. "
    "Lending policy questions are not supported yet."
)


def render(
    officer_email: str, token: str, header: Callable[[], None] = loana.no_header,
    officer_name: str = "",
) -> None:
    """`header` draws the page bar: full width on the landing, in the main column after."""
    _keep_widget_state()

    header()
    customer_id = st.session_state.get(CUSTOMER_KEY)
    view = nav.current()
    if view == nav.CUSTOMER and customer_id:
        _record_page(customer_id, officer_email, token)
        loana.dock(lambda: _assistant(customer_id))
    elif view == nav.PREDICTION and customer_id:
        _prediction_page(customer_id, officer_email)
        loana.dock(lambda: _assistant(customer_id))
    elif view == nav.CONVERSATION and st.session_state.get(CONVERSATION_KEY):
        conversation_id = st.session_state[CONVERSATION_KEY]
        _conversation_page(conversation_id, officer_email)
        loana.dock(lambda: _assistant(conversation_id))
    elif view == nav.CHAT:
        _messages(_all_requests(), officer_email)
    else:
        _landing(officer_email, officer_name)


def close_customer() -> None:
    """Drop the opened customer from the drawer; leave their page if it is showing."""
    if nav.current() in (nav.CUSTOMER, nav.PREDICTION):
        st.session_state[nav.VIEW_KEY] = nav.DASHBOARD
    st.session_state[CUSTOMER_KEY] = None


def close_conversation() -> None:
    """Drop the opened review from the drawer; back to Officer reviews if it is showing."""
    if nav.current() == nav.CONVERSATION:
        st.session_state[nav.VIEW_KEY] = nav.CHAT
    st.session_state[CONVERSATION_KEY] = None


def conversation_label() -> str | None:
    """The opened conversation's customer name for the drawer, or None."""
    return _name_of(st.session_state.get(CONVERSATION_KEY))


def waiting_count() -> int:
    """Customers waiting for a decision: the red bubble beside Officer reviews in the drawer."""
    return len(review.conversations(_all_requests()))


def customer_label() -> str | None:
    """The opened customer's name for the drawer, or None when nobody is open."""
    return _name_of(st.session_state.get(CUSTOMER_KEY))


def _name_of(customer_id: str | None) -> str | None:
    if not customer_id:
        return None
    try:
        detail = api.customer_detail(customer_id)
    except api.ServiceUnavailable:
        return "Customer"
    return detail.get("full_name") or f"Customer {detail['account_number']}"


def _keep_widget_state() -> None:
    for key in list(st.session_state):
        if key.startswith(KEPT_PREFIXES):
            st.session_state[key] = st.session_state[key]


# --- the selected customer ------------------------------------------------


def _select(customer_id: str | None) -> None:
    """Open this customer, or None to go back to the landing page.

    Used as a button callback, which is the only way a customer is ever opened.
    """
    st.session_state[CUSTOMER_KEY] = customer_id
    st.session_state.pop(NOTICE_KEY, None)
    # Navigate even when already selected: the customer can be open in the drawer
    # while another view (e.g. their conversation) is showing.
    if customer_id:
        nav.go(nav.CUSTOMER)


def _open_conversation(customer_id: str) -> None:
    st.session_state[CONVERSATION_KEY] = customer_id
    st.session_state.pop(NOTICE_KEY, None)
    nav.go(nav.CONVERSATION)


# --- review requests ------------------------------------------------------


def _all_requests() -> list[dict]:
    """Every open cardholder message; empty when the service is down, so the page still draws."""
    try:
        return api.review_requests()
    except api.ServiceUnavailable:
        return []


def _requests(requests: list[dict], officer_email: str) -> None:
    waiting = review.conversations(requests)
    st.subheader("Customers asking for a human review")
    st.caption(
        f"{len(waiting)} waiting · {review.pending_count(requests)} unread requests. "
        "A review is when a person decides whether the model's rating should stand."
    )
    if not waiting:
        st.info("No customers are waiting for a review.")
        return

    now = datetime.now(timezone.utc)
    for item in waiting[:REQUEST_CARDS]:
        _request_card(item, _band(item["customer_id"]), now, officer_email)
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


def _approve(customer_id: str, officer_email: str) -> None:
    """Accept the request and open the conversation."""
    try:
        api.approve_review(customer_id, officer_email)
    except api.ServiceUnavailable as exception:
        st.session_state[NOTICE_KEY] = {"customer_id": customer_id, "ok": False, "text": str(exception)}
        return
    _open_conversation(customer_id)


def _reject(customer_id: str, officer_email: str) -> None:
    """Decline the request; the card leaves the list."""
    try:
        api.reject_review(customer_id, officer_email)
    except api.ServiceUnavailable as exception:
        st.session_state[NOTICE_KEY] = {"customer_id": customer_id, "ok": False, "text": str(exception)}


def _request_card(item: dict, band: str | None, now: datetime, officer_email: str) -> None:
    card = review.card_data({**item, "reason": item["latest"]}, band, now)
    unread = item["unread"]
    with st.container(border=True, key=f"request_card_{item['request_id']}"):
        show(card_head(
            card["name"],
            f"Account {card['account']} · waiting {card['waiting']} · "
            f"{unread} unread request{'' if unread == 1 else 's'}",
            card["band"],
            lift=True,
        ))
        show(quote(card["note"]))
        if card["is_truncated"]:
            with st.expander("Read the full note"):
                show(quote(card["full_note"]))
        # Pending opens by accepting; in review is already accepted, so it just opens.
        pending = item["status"] == "pending"
        open_column, reject_column, _ = st.columns([1.6, 1.6, 4])
        open_column.button(
            "Open review",
            key=f"{'approve' if pending else 'review'}_{item['request_id']}",
            on_click=_approve if pending else _open_conversation,
            args=(item["customer_id"], officer_email) if pending else (item["customer_id"],),
        )
        reject_column.button(
            "Reject request", key=f"reject_{item['request_id']}",
            on_click=_reject, args=(item["customer_id"], officer_email),
        )


# --- the landing page -----------------------------------------------------


def _landing(officer_email: str, officer_name: str) -> None:
    """Nobody is open: Loana's briefing, the command bar, then the worklist."""
    requests = _all_requests()
    _briefing(officer_name, len(review.conversations(requests)))
    _command_bar(officer_email)
    _worklist(officer_email, review.unread_by_customer(requests))


def _greeting(officer_name: str) -> str:
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    first = officer_name.split()[0] if officer_name.strip() else ""
    return f"Good {part}, {first}." if first else f"Good {part}."


def _typing(text: str) -> Iterator[str]:
    for word in text.split(" "):
        yield word + " "
        time.sleep(TYPING_DELAY)


def _say(text: str, fresh: bool) -> None:
    """Loana's words: typed out when they are new, plain once they have been read."""
    if fresh:
        st.write_stream(_typing(markdown_safe(text)))
    else:
        st.markdown(markdown_safe(text))


def _briefing(officer_name: str, waiting: int) -> None:
    """Loana opens the page: what is waiting, a place to start, one-click cohorts.

    The text comes from the backend, written from the counts, so it cannot claim
    a number the worklist does not hold.
    """
    with st.container(key=BRIEFING_KEY):
        loana.header("Triage assistant")
        st.markdown(f"### {_greeting(officer_name)}")
        try:
            brief = api.briefing()
        except api.ServiceUnavailable:
            st.markdown(
                "I can't reach the worklist right now. Once the service is back I'll "
                "tell you who to look at first."
            )
            return

        text = brief["text"]
        if waiting:
            text += (
                f" **{waiting}** {'customer is' if waiting == 1 else 'customers are'} "
                "also waiting for a human review in Officer reviews."
            )
        _say(text, fresh=not st.session_state.get(BRIEFED_KEY))
        st.session_state[BRIEFED_KEY] = True

        with st.container(horizontal=True, key="officer_suggestions"):
            for number, cohort in enumerate(brief["cohorts"]):
                st.button(
                    f"{cohort['label']} · {cohort['total']:,}",
                    key=f"officer_quick_{number}", disabled=not cohort["total"],
                    on_click=_quick_cohort, args=(cohort["filter"],),
                )
            lead = brief.get("lead")
            if lead:
                who = lead.get("full_name") or f"account {lead['account_number']}"
                st.button(
                    f"Open {who}'s dashboard →", key="officer_lead", type="primary",
                    on_click=_select, args=(lead["customer_id"],),
                )
            if waiting:
                st.button(
                    "Open Officer reviews", key="officer_open_messages",
                    on_click=nav.go, args=(nav.CHAT,),
                )


# --- the command bar --------------------------------------------------------


def _is_account(text: str) -> bool:
    return re.fullmatch(r"[A-Za-z]?-?\d+", text) is not None


def _looks_like_lookup(text: str) -> bool:
    """An account number, or a short phrase with no words that describe a group."""
    if _is_account(text):
        return True
    words = text.lower().split()
    return len(words) <= 3 and not (set(words) & COHORT_WORDS) and not re.search(r"\d", text)


def _command_bar(officer_email: str) -> None:
    """One box: a name or account finds the customer, anything else goes to Loana."""
    # Inside a container, so the chat input sits here rather than pinned to the foot.
    with st.container(key="officer_command_bar"):
        asked = st.chat_input(COMMAND_PLACEHOLDER, key=COMMAND_KEY)
    if asked and asked.strip():
        _handle(asked.strip(), officer_email)
    _exchange()
    _search_results()


def _handle(asked: str, officer_email: str) -> None:
    if _looks_like_lookup(asked):
        try:
            matches = api.search_customers(asked)
        except api.ServiceUnavailable as exception:
            api.guard(exception)
            return
        # A name nobody has goes on to Loana; an account number nobody has is a miss.
        if matches or _is_account(asked):
            st.session_state[SEARCH_KEY] = asked
            count = len(matches)
            # Search returns at most SEARCH_LIMIT, so a full page may hide more.
            found = (
                f"Here are the first {count} customers" if count >= SEARCH_LIMIT
                else f"Found {count} customer{'' if count == 1 else 's'}"
            )
            reply = (
                f"{found} matching “{asked}”. Pick one to open their dashboard."
                if matches else f"No customer matches “{asked}”."
            )
            st.session_state[EXCHANGE_KEY] = {"asked": asked, "reply": reply, "fresh": True}
            return
    st.session_state.pop(SEARCH_KEY, None)
    _ask_loana(asked, officer_email)


def _ask_loana(asked: str, officer_email: str) -> None:
    """Loana fills a filter; the backend checks it and writes her reply from it,
    so nothing she says can name a customer or a risk the data does not back."""
    messages = st.session_state.setdefault(TRIAGE_CHAT_KEY, [])
    sent = [
        *({"role": m["role"], "content": m["content"]} for m in messages),
        {"role": "user", "content": asked},
    ]
    with st.spinner("LOANA is reading the worklist..."):
        try:
            turn = api.triage(sent, asked_by=officer_email)
        except api.ServiceUnavailable:
            st.session_state[EXCHANGE_KEY] = {
                "asked": asked, "reply": "I'm not connected yet. Try again in a moment.",
                "fresh": True,
            }
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
    st.session_state[EXCHANGE_KEY] = {
        "asked": asked, "reply": reply, "note": turn.get("note", ""), "fresh": True,
    }


def _exchange() -> None:
    """The officer's last request and Loana's answer, under the command bar."""
    last = st.session_state.get(EXCHANGE_KEY)
    if not last:
        return
    with st.container(key="officer_reply"):
        with st.chat_message("user"):
            st.markdown(last["asked"])
        with st.chat_message("assistant", avatar=str(loana.AVATAR)):
            _say(last["reply"], fresh=last.pop("fresh", False))
            if last.get("note"):
                st.caption(last["note"])


def _search_results() -> None:
    """Matches as buttons. Nothing opens until one is clicked, even a lone match."""
    typed = st.session_state.get(SEARCH_KEY)
    if not typed:
        return
    try:
        matches = api.search_customers(typed)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return
    with st.container(horizontal=True, key="officer_matches"):
        for match in matches:
            st.button(
                f"{_label(match)} →", key=f"officer_pick_{match['customer_id']}",
                on_click=_select, args=(match["customer_id"],),
            )



def _messages(requests: list[dict], officer_email: str) -> None:
    """Officer reviews view: every request waiting on the officer."""
    _requests(requests, officer_email)


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
    prediction.entry_card(prediction.officer_summary(detail))
    if request:
        _plan(detail, token)
    _figures(detail)
    st.divider()
    _history(detail)


def _prediction_page(customer_id: str, officer_email: str) -> None:
    """The model's call on the next payment for the opened customer, in full."""
    try:
        detail = api.customer_detail(customer_id)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return
    _heading(detail)
    prediction.officer_page(detail, officer_email)

    if request:
        st.divider()
        _evidence(detail, request)
        _plan(detail, token)


# --- the conversation -------------------------------------------------------


def _conversation_page(customer_id: str, officer_email: str) -> None:
    """The officer's side of the review, in the same card the cardholder sees.

    An officer's reply is a decision with a note, so the form issues a decision.
    The cardholder reads the same record.
    """
    stylesheet("chat_page.css")
    nav.back_button("Back to worklist")
    try:
        detail = api.customer_detail(customer_id)
        entries = api.thread(customer_id)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return

    name = detail.get("full_name") or f"Customer {detail['account_number']}"
    open_request = _open_request_for(customer_id) is not None
    closed = review.is_closed(entries)
    badge = thread.status_chip(entries)
    with st.container(key="chat_card"):
        with st.container(key="chat_head"):
            title, status = st.columns([5, 1.4], vertical_alignment="center")
            # The name pill is the way to the customer's dashboard; the hint sits beside it.
            with title.container(horizontal=True, vertical_alignment="center"):
                st.button(
                    name, key="officer_conversation_dashboard", type="tertiary",
                    on_click=_select, args=(customer_id,),
                )
                st.markdown(
                    '<span class="chat-pill-hint">Click the name to see their dashboard</span>',
                    unsafe_allow_html=True,
                )
            status.markdown(f'<div class="chat-head-status">{badge}</div>', unsafe_allow_html=True)
        with st.container(key="chat_messages"):
            if entries:
                thread.draw(entries, "officer")
            else:
                show(
                    '<div class="chat-empty"><div class="chat-empty-icon">&#128196;</div>'
                    "<b>No review requests yet</b>"
                    f"<span>A decision you issue reaches {name} under Officer reviews, "
                    "recorded on the account.</span></div>"
                )
            if closed:
                show(
                    '<div class="chat-closed"><b>This review is complete</b>'
                    "<span>It reopens when the customer submits a new request.</span></div>"
                )

        with st.container(key="chat_composer"):
            level, box, send = st.columns([1.4, 6, 1.3], vertical_alignment="bottom")
            # Widgets rather than a form: a form's unsent text never reaches session
            # state, so it could not survive a trip to another view.
            band_key = f"officer_band_{customer_id}"
            st.session_state.setdefault(band_key, review.current_band(detail))
            level.selectbox("Risk level", review.RISK_BANDS, key=band_key)
            box.text_area(
                "Decision note", key=f"officer_note_{customer_id}",
                placeholder=f"Explain the decision to {name}, in words they will understand...",
                label_visibility="collapsed", height=80,
            )
            send.button(
                "Issue decision", type="primary", key=f"officer_record_{customer_id}", width="stretch",
                on_click=_record,
                args=(customer_id, officer_email, detail.get("full_name"),
                      detail["account_number"], open_request),
            )

    notice = st.session_state.get(NOTICE_KEY)
    if notice and notice["customer_id"] == customer_id:
        (st.success if notice["ok"] else st.error)(notice["text"])


# --- the review itself ----------------------------------------------------


def _unread_alert(request: dict) -> None:
    """The customer is waiting: say so above everything else on their record."""
    unread = request["unread"]
    waited = review.waiting_text(request["requested_at"], datetime.now(timezone.utc))
    show(
        f'<div class="inbox-alert"><span class="inbox-badge">'
        f"{review.capped(unread, UNREAD_CAP)}</span>"
        f"<div><b>🔔 {unread} unanswered request{'' if unread == 1 else 's'} from this "
        f"customer</b> · waiting {waited}. Issue a decision from their review in "
        "Officer reviews.</div></div>"
    )


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
            st.markdown(f"- {markdown_safe(step['description'])}")
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
            ("Credit limit", money(detail['credit_limit']), "The agreed limit."),
            ("Balance", money(detail['current_balance']), "Owed this month."),
            ("Last payment", money(detail['last_payment']), "Most recent payment."),
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
                    "Bill": money(entry['bill_amount']),
                    "Paid": money(entry['amount_paid']),
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


def _record(
    customer_id: str, officer_email: str, name: str | None,
    account_number: str, from_request: bool,
) -> None:
    """Save the reply as a decision. A callback, so it can clear the note that was just sent."""
    action = review.DEFAULT_ACTION
    band = st.session_state.get(f"officer_band_{customer_id}")
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
        api.decide(customer_id, officer_email, action, justification, band)
    except api.ServiceUnavailable as exception:
        notify(False, str(exception))
        return

    st.session_state[note_key] = ""
    who = f"**{name}** (account {account_number})" if name else f"account **{account_number}**"
    outcome = (
        "Decision issued and review closed" if from_request else "Decision issued"
    )
    notify(True, f"{outcome} for {who}: risk level set to {band}.")


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
                st.markdown(markdown_safe(message["content"]))
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


# --- the worklist ------------------------------------------------------------


def _show_cohort(page: dict, criteria: dict) -> None:
    """Make this cohort the worklist, forgetting the previous one's ticks."""
    st.session_state[COHORT_KEY] = page
    st.session_state[COHORT_FILTER_KEY] = criteria
    st.session_state.pop(JUST_SCORED_KEY, None)
    _clear_ticks()


def _clear_ticks() -> None:
    st.session_state[EDITOR_VERSION_KEY] = st.session_state.get(EDITOR_VERSION_KEY, 0) + 1


def _quick_cohort(criteria: dict) -> None:
    """Button callback: pull a cohort without asking Loana."""
    try:
        _show_cohort(api.cohort(criteria), criteria)
    except api.ServiceUnavailable as exception:
        st.session_state[COHORT_ERROR_KEY] = str(exception)


def _chip_label(field: str, value) -> str:
    return {
        "min_months_in_arrears": lambda: f"{value}+ months in arrears",
        "trajectory": lambda: f"Trend: {value}",
        "min_months_with_no_payment": lambda: f"No payment in {value}+ months",
        "min_utilisation": lambda: f"Using {value:.0%}+ of limit",
        "min_credit_limit": lambda: f"Limit at least {money(value)}",
        "max_credit_limit": lambda: f"Limit at most {money(value)}",
        "not_scored": lambda: "No officer score yet",
        "scored_band": lambda: f"Scored {value} risk",
    }[field]()


def _chips(criteria: dict) -> None:
    """The filter behind the worklist, one chip per condition; unticking one drops it."""
    active = [field for field in CHIP_FIELDS if criteria.get(field) not in (None, False, 0)]
    if not active:
        return
    key = f"{CHIPS_KEY}_{st.session_state.get(EDITOR_VERSION_KEY, 0)}"
    st.pills(
        "LOANA's filter", active, selection_mode="multi", default=active, key=key,
        format_func=lambda field: _chip_label(field, criteria[field]),
        on_change=_drop_chips, args=(key,),
        help="Click a condition to drop it and widen the group.",
    )


def _drop_chips(key: str) -> None:
    kept = set(st.session_state.get(key) or [])
    criteria = {
        field: value
        for field, value in st.session_state[COHORT_FILTER_KEY].items()
        if field not in CHIP_FIELDS or field in kept
    }
    _quick_cohort(criteria)


def _tick_key(customer_id: str) -> str:
    return f"{TICK_PREFIX}{st.session_state.get(EDITOR_VERSION_KEY, 0)}_{customer_id}"


def _tick_all(rows: list[dict], key: str) -> None:
    for row in rows:
        st.session_state[_tick_key(row["customer_id"])] = st.session_state[key]


def _worklist(officer_email: str, unread: dict[str, int]) -> None:
    """The cohort Loana or a suggestion pulled in, to tick, predict and open."""
    st.subheader("Worklist")
    if st.session_state.pop(COHORT_ERROR_KEY, None):
        st.warning("Could not load that cohort. Is the API running?")

    page = st.session_state.get(COHORT_KEY)
    if not page:
        st.info(EMPTY_WORKLIST)
        return

    _just_scored()
    _chips(st.session_state.get(COHORT_FILTER_KEY) or {})
    st.caption(f"{markdown_safe(page['description'])} Showing {len(page['rows'])} of {page['total']:,}.")
    rows = page["rows"]
    if not rows:
        st.info("Nobody matches. Drop a condition above to widen the group.")
        return

    with st.container(key="officer_worklist"):
        _worklist_header(rows)
        for row in rows:
            _worklist_row(row, unread.get(row["customer_id"], 0))
    chosen = [row for row in rows if st.session_state.get(_tick_key(row["customer_id"]))]

    too_many = len(chosen) > MAX_PREDICT
    if st.button(
        f"Predict selected ({len(chosen)})", type="primary", width="stretch",
        disabled=not chosen or too_many, key="officer_predict_selected",
    ):
        _predict_selected(chosen, officer_email)
    if too_many:
        st.caption(f"Up to {MAX_PREDICT} at a time.")


def _worklist_header(rows: list[dict]) -> None:
    columns = st.columns(WORKLIST_WIDTHS, vertical_alignment="center")
    key = f"{TICK_PREFIX}all_{st.session_state.get(EDITOR_VERSION_KEY, 0)}"
    columns[0].checkbox(
        "Select all", key=key, label_visibility="collapsed",
        on_change=_tick_all, args=(rows, key),
    )
    for column, heading in zip(columns[1:], WORKLIST_HEADINGS[1:]):
        if heading:
            column.markdown(f'<span class="worklist-heading">{heading}</span>', unsafe_allow_html=True)


def _worklist_row(row: dict, unread: int = 0) -> None:
    customer_id = row["customer_id"]
    band = row["last_risk_band"]
    utilisation = row["utilisation_now"]
    name = row["full_name"] or "-"
    # Customers waiting on a reply stand out in whatever cohort they land in.
    if unread:
        name += f" &nbsp;🔔 {review.capped(unread, UNREAD_CAP)}"
    if band:
        name += f" &nbsp;{BAND_ICONS[band]} {band}"

    with st.container(key=f"worklist_row_{customer_id}"):
        columns = st.columns(WORKLIST_WIDTHS, vertical_alignment="center")
        columns[0].checkbox(
            f"Select {row['account_number']}", key=_tick_key(customer_id),
            label_visibility="collapsed",
        )
        for column, value in zip(columns[1:-1], (
            row["account_number"],
            name,
            f"{row['months_in_arrears']} of 6",
            (row["trajectory"] or "-").capitalize(),
            "-" if utilisation is None else f"{utilisation:.0%}",
            f"{row['months_with_no_payment']} of 6",
            markdown_safe(money(row['current_balance'])),
        )):
            column.markdown(value)
        columns[-1].button(
            "View user dashboard →", key=f"officer_open_{customer_id}", type="tertiary",
            on_click=_select, args=(customer_id,),
        )


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
