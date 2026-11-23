"""Cardholder view: an early warning the customer can act on.

Shows no probability and no threshold, because the API does not send them. That
enforcement is the backend's (`worklist.cardholder_view`), not this screen's: a
page that merely declined to draw the number would still have received it. The
same holds for *whose* account this is: the screen is handed a session token and
a customer id, and the API refuses the pairing unless the signed-in caller is
that cardholder or a member of staff.

Telling someone they are predicted to default is distressing, can be
self-fulfilling, and commits the lender to specific reasons the moment it acts.
Pre-delinquency contact is framed as assistance.

The page stays visual: the figures, the bill-against-payment timeline, and the
human review request page (ui/thread.py). Reasons and what would help are
Loana's to explain, on request, in a floating chat in the corner (ui/loana.py).
Loana's wording and the repayment history it draws on come from the API.
"""
from collections.abc import Callable
from html import escape

import streamlit as st

import api
import review
import theme
from money import CURRENCY, markdown_safe, money, money_or_none, to_usd
from ui import insights, loana, nav, thread
from ui.components import show, stylesheet
from views import prediction

# Features a cardholder can act on this month (read by the officer screen). A reason
# they cannot change is not advice, it is an accusation.
ACTIONABLE = {
    "PAY_1": "Bring the account up to date before the next statement.",
    "PAY_2": "Clearing the older arrears first stops them compounding.",
    "pay_max": "Catching up the worst month has the largest effect.",
    "pay_mean": "Steadily catching up across the months helps most.",
    "pay_months_late": "Each month brought up to date lowers the risk.",
    "months_zero_payment": "Making any payment is much better than none.",
    "PAY_AMT1": "Paying more than the minimum this month helps most.",
    "remaining_credit": "Keeping some credit free gives you room to absorb a bill.",
    "util_1": "Bringing the balance below about 70% of the limit helps.",
    "util_mean": "Lowering the balance over a few months helps steadily.",
    "util_max": "Try not to run the card close to its limit.",
    "repay_ratio_1": "Paying a larger share of the statement each month helps.",
    "repay_ratio_min": "Avoid months where only a small part of the bill is paid.",
    "pay_to_bill_gap": "Closing the gap between what you spend and what you repay helps.",
    # Concept reasons. Account size, recent direction and the limit are not on the list.
    "repayment_status": "Bring the account up to date before the next statement.",
    "share_paid": "Paying a larger share of each bill helps most.",
    "carrying_balance": "Clearing the bill in full, rather than carrying part over, helps.",
    "credit_in_use": "Keeping the balance well below the limit gives you room to absorb a bill.",
}

REVIEW_COMMENT_MAX = 2000

# Starters for an empty conversation; clicking one fills the box, it never sends.
QUICK_PROMPTS = (
    "I would like my latest statement reviewed.",
    "I would like to discuss my repayment options.",
    "I believe there is an error on my account.",
)

# Fixed questions for the two preset buttons. They go through the same chat endpoint,
# guards and log as anything typed.
PRESETS = {
    "💡 Why flagged?": "Why was my account flagged?",
    "🎯 Next steps": "What can I do this month?",
}

NOT_SCORED = (
    "Your account does not have six full months of history yet, so it cannot be "
    "scored and there is no dashboard to show."
)
PREDICT_PROBLEM_KEY = "predict_problem"

GREETING = (
    "Hi, I'm Loana. Ask me about your account, or pick one of the questions below."
)
DISCLAIMER = "Loana answers from your account's own data and is not financial advice."

CONSENT = """
Your account is checked each month using your billing and repayment history for
the last six months.

The check never uses your sex, age, education or marital status. Those are
excluded from the model entirely, and are kept only so the bank can test the
system for fairness.

No decision about your account is made automatically. A member of staff reviews
anything that needs action, and you can ask for a human review of any decision.
"""


def render(
    token: str, customer_id: str, header: Callable[[], None] = loana.no_header
) -> None:
    """`header` draws the page bar at the top of the main column."""
    try:
        account = api.cardholder(customer_id, token)
    except api.ServiceUnavailable as exception:
        header()
        if exception.code == "RESOURCE_NOT_FOUND":
            # Every stored customer is scorable; only one without six full months lands here.
            st.info(NOT_SCORED)
            st.stop()
        api.guard(exception)
        return

    header()
    if nav.current() == nav.CHAT:
        _chat_view(token, customer_id)
        return
    if nav.current() == nav.PREDICTION:
        _prediction_view(token, customer_id, account)
        loana.dock(lambda: _copilot(token, customer_id))
        return
    if not account.get("predicted", True):
        _unpredicted_dashboard(token, customer_id, account)
        return
    _dashboard(account)
    loana.dock(lambda: _copilot(token, customer_id))


def _dashboard(account: dict) -> None:
    """The account itself: the banner, the figures, the history and what to do about it."""
    stylesheet("dashboard.css")
    needs_attention = account["needs_attention"]
    _officer_decision(account.get("officer_decision"))
    _banner(account)

    _account(account)
    insights.render(account["history"], account["credit_limit"])
    _ask_loana(needs_attention)
    _consent()


def _unpredicted_dashboard(token: str, customer_id: str, account: dict) -> None:
    """Figures and history only; no band until a prediction is run."""
    stylesheet("dashboard.css")
    _officer_decision(account.get("officer_decision"))
    _account(account)
    insights.render(account["history"], account["credit_limit"])


def _prediction_view(token: str, customer_id: str, account: dict) -> None:
    """The Risk Prediction page: the band and the reasons, never the probability."""
    stylesheet("dashboard.css")
    _officer_decision(account.get("officer_decision"))
    prediction.cardholder_page(
        account, on_predict=lambda: _predict(token, customer_id),
        problem=st.session_state.pop(PREDICT_PROBLEM_KEY, None),
    )


def _predict(token: str, customer_id: str) -> None:
    try:
        api.request_prediction(customer_id, token)
    except api.ServiceUnavailable as exception:
        st.session_state[PREDICT_PROBLEM_KEY] = str(exception)


def _chat_view(token: str, customer_id: str) -> None:
    """The human review request page, as the whole workspace."""
    stylesheet("chat_page.css")
    nav.back_button()
    _human_review(token, customer_id)


def _officer_decision(decision: dict | None) -> None:
    """Who set the account's risk level, when an officer has."""
    if not decision:
        return
    when = escape(review.format_time(decision["decided_at"]))
    st.markdown(
        '<div class="account-alert account-alert--good">'
        '<div class="account-alert-title">Reviewed by a loan officer</div>'
        f'<p>Loan officer <b>{escape(decision["officer_name"])}</b> reviewed your account on '
        f'{when} and decided you belong to <b>{escape(decision["risk_band"])} risk</b>.</p></div>',
        unsafe_allow_html=True,
    )


# Months behind at which an account is past "early warning" and into recovery.
SERIOUSLY_BEHIND_MONTHS = 3
# Payments under this share of what was billed are not keeping pace with the bills.
LOW_REPAYMENT_SHARE = 0.5


def banner_copy(account: dict) -> tuple[str, str, str]:
    """(tone, title, body) for the status callout, worded to match the account's own record.

    The model's flag says whether to warn; the history says how. An account behind
    every month is told so plainly, not given an "early heads-up".
    """
    history = account.get("history") or []
    behind = account["months_in_arrears"]
    latest = history[0]["payment_status"] if history else 0
    billed = sum(month["bill_amount"] for month in history)
    paid = sum(month["amount_paid"] for month in history)
    share = paid / billed if billed > 0 else None
    window = f"{behind} of the last {len(history) or 6} months"

    if not account["needs_attention"]:
        if behind == 0:
            return "good", "Your account looks on track", (
                "Nothing in your recent repayment history suggests you will have "
                "difficulty with your next payment."
            )
        return "good", "Your account is on track", (
            f"You were behind in {window}, but your recent payments suggest you "
            "should manage your next one. Staying up to date keeps it that way."
        )

    tone = "high" if account["risk_band"] == "High" else "medium"
    if behind >= SERIOUSLY_BEHIND_MONTHS or latest >= SERIOUSLY_BEHIND_MONTHS:
        overdue = (
            f" Your latest payment is {latest} months overdue." if latest > 1 else ""
        )
        shortfall = (
            f" Over that time your payments covered {share:.0%} of what was billed, "
            "so the balance has kept growing."
            if share is not None and share < LOW_REPAYMENT_SHARE else ""
        )
        return tone, "Your account is seriously behind", (
            f"Your account has been behind on payments in {window}.{overdue}{shortfall} "
            "This is not a decision, but catching up on your own is now unlikely. "
            "Requesting a review lets a loan officer discuss a repayment plan with you."
        )
    if behind > 0:
        return tone, "Your account has started to fall behind", (
            f"You were behind on payments in {window}. Catching up before the next "
            "statement is the most effective step, and it is still within reach. "
            "This is not a decision."
        )
    return tone, "Your next payment may be at risk", (
        "Your payments are up to date, but how your balance and payments have moved "
        "over the last six months suggests your next payment could be difficult. "
        "This is not a decision, just an early heads-up while there is still time to act."
    )


def _banner(account: dict) -> None:
    """Status callout: rose for a flagged high band, amber for other flags, mint otherwise."""
    tone, title, body = banner_copy(account)
    st.markdown(
        f'<div class="account-alert account-alert--{tone}">'
        f'<div class="account-alert-title">{title}</div><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def _human_review(token: str, customer_id: str) -> None:
    """Request a human review: the case record, then a form to submit a request.

    Offered to everyone, flagged or not. A decision completes the review;
    submitting again opens a new one.
    """
    try:
        entries = api.cardholder_thread(customer_id, token)
    except api.ServiceUnavailable:
        st.caption("Review requests are unavailable right now.")
        return

    closed = review.is_closed(entries)
    with st.container(key="chat_card"):
        show(
            f'<div class="chat-head"><b>Request a human review</b>{thread.status_chip(entries)}</div>'
        )
        with st.container(key="chat_messages"):
            if entries:
                thread.draw(entries, "cardholder")
            else:
                _empty_state(customer_id)
            if closed:
                show(
                    '<div class="chat-closed"><b>This review is complete</b>'
                    "<span>Submit a new request below if you would like another review.</span></div>"
                )

        with st.container(key="chat_composer"):
            st.text_area(
                "What would you like reviewed?",
                key=f"review_comment_{customer_id}",
                placeholder="Describe what you would like a loan officer to look at.",
                max_chars=REVIEW_COMMENT_MAX,
                height=80,
            )
            st.button(
                "Submit request", key=f"review_send_{customer_id}", type="primary",
                on_click=_send, args=(token, customer_id),
            )
            st.caption(
                "A loan officer will review your request and reply on this page. "
                "No decision is made automatically."
            )
        problem = st.session_state.pop(f"review_problem_{customer_id}", None)
        if problem:
            st.error(problem)


def _empty_state(customer_id: str) -> None:
    """Icon, what happens next and a few starters, before anything has been submitted."""
    show(
        '<div class="chat-empty"><div class="chat-empty-icon">&#128196;</div>'
        "<b>No review requests yet</b>"
        "<span>Ask a loan officer to look again at your account or at a rating. "
        "Your request goes to the officer reviewing your account, and their decision "
        "will appear here.</span></div>"
    )
    with st.container(key="chat_chips"):
        for index, prompt in enumerate(QUICK_PROMPTS):
            st.button(
                prompt, key=f"chip_{customer_id}_{index}",
                on_click=_fill, args=(customer_id, prompt),
            )


def _fill(customer_id: str, prompt: str) -> None:
    st.session_state[f"review_comment_{customer_id}"] = prompt


def _send(token: str, customer_id: str) -> None:
    """A callback, so the box can be cleared once the request is submitted."""
    key = f"review_comment_{customer_id}"
    comment = st.session_state.get(key, "").strip()
    if not comment:
        st.session_state[f"review_problem_{customer_id}"] = (
            "Please describe what you would like reviewed."
        )
        return
    try:
        api.request_review(customer_id, token, comment)
    except api.ServiceUnavailable as exception:
        st.session_state[f"review_problem_{customer_id}"] = str(exception)
        return
    st.session_state[key] = ""


def _account(account: dict) -> None:
    for column, (label, value) in zip(
        st.columns(4),
        [
            ("Credit limit", money(account['credit_limit'])),
            ("Latest statement balance", money(account['current_balance'])),
            ("Last payment", money_or_none(account['last_payment'])),
            ("Months behind", f"{account['months_in_arrears']} of 6"),
        ],
    ):
        column.metric(label, value)


def _ask_loana(needs_attention: bool) -> None:
    """Why, and what would help, are answered in the chat rather than written out here."""
    question = (
        "what would help most this month" if needs_attention else "what keeps it on track"
    )
    st.markdown(
        f'<div class="loana-hint">Want to know why, or {question}? '
        "<b>Ask Loana</b> in the corner. She explains it with your own figures.</div>",
        unsafe_allow_html=True,
    )


def _copilot(token: str, customer_id: str) -> None:
    """Loana AI: questions about this account, answered from the account's own facts.

    The conversation lives in session state and is sent whole each turn; the API
    keeps none. Every reply's wording comes from the API, and only after the customer
    asks, because each reply can cost the API an LLM call.
    """
    key = f"chat_{customer_id}"
    # No API call until the customer asks: the greeting is local, and the real
    # disclaimer replaces the placeholder with the first answer.
    conversation = st.session_state.setdefault(
        key,
        {
            "messages": [{"role": "assistant", "content": GREETING, "local": True}],
            "disclaimer": DISCLAIMER,
        },
    )

    # Messages scroll in their own box; presets and the input stay under it.
    chat = loana.chat_area()
    with chat:
        for message in conversation["messages"]:
            _bubble(message)

    asked = None
    for column, (label, question) in zip(st.columns(len(PRESETS)), PRESETS.items()):
        if column.button(label, key=f"preset_{customer_id}_{label}", width="stretch"):
            asked = question

    asked = st.chat_input("Ask why, or what would help") or asked or loana.take_pending()
    st.caption(conversation["disclaimer"])
    if not asked:
        return

    conversation["messages"].append({"role": "user", "content": asked})
    sent = [
        {"role": m["role"], "content": m["content"]}
        for m in conversation["messages"]
        if not m.get("local")
    ]
    with chat:
        _bubble(conversation["messages"][-1])
        with st.spinner("Thinking about your account..."):
            try:
                turn = api.chat(customer_id, token, sent)
                conversation["disclaimer"] = turn.get("disclaimer") or DISCLAIMER
                # The API's fallback note is for logs and officers; a cardholder is not shown it.
                reply = {"role": "assistant", "content": turn["reply"]}
            except api.ServiceUnavailable:
                reply = {
                    "role": "assistant",
                    "content": "I could not reach the service just now. Please try again.",
                }
    conversation["messages"].append(reply)
    st.rerun()


def _bubble(message: dict) -> None:
    with st.chat_message(message["role"]):
        # Escaped, or a pair of dollar amounts in one reply renders as a formula.
        st.markdown(markdown_safe(message["content"]))


def _consent() -> None:
    with st.expander("How this check works, and your rights"):
        st.markdown(CONSENT)
        st.checkbox("Send me an early warning if my account looks at risk", value=True)
        st.caption("Preference not yet saved.")
