"""Cardholder view: an early warning the customer can act on.

Shows no probability and no threshold, because the API does not send them. That
enforcement is the backend's (`worklist.cardholder_view`), not this screen's: a
page that merely declined to draw the number would still have received it. The
same holds for *whose* account this is: the screen is handed a session token and
a customer id, and the API refuses the pairing unless the signed-in caller is
that cardholder or a member of staff.

Telling someone they are predicted to default is distressing, can be
self-fulfilling, and commits the lender to specific reasons the moment it acts.
Pre-delinquency contact is framed as assistance, so this screen says what to do.

The account at full width, with Loana AI (the chat copilot) as a floating chat in the
corner (ui/loana.py). Loana's wording, and the repayment history it draws on, come
from the API.
"""
from collections.abc import Callable

import altair as alt
import pandas as pd
import streamlit as st

import api
import review
import theme
from ui import loana

# Features a cardholder can act on this month. A reason they cannot change is
# not advice, it is an accusation.
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
}

SHOWN_ADVICE = 3
REVIEW_COMMENT_MAX = 2000

# Fixed questions for the two preset buttons. They go through the same chat endpoint,
# guards and log as anything typed.
PRESETS = {
    "💡 Why flagged?": "Why was my account flagged?",
    "🎯 Next steps": "What can I do this month?",
}

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
        api.guard(exception)
        return

    header()
    _main(account, token, customer_id)
    loana.dock(lambda: _copilot(token, customer_id))


def _main(account: dict, token: str, customer_id: str) -> None:
    """The account itself: the banner, the figures, the history and what to do about it."""
    needs_attention = account["needs_attention"]
    if needs_attention:
        _warning()
    else:
        _reassurance(account)

    _account(account)
    _history(account["history"])

    if needs_attention:
        _what_to_do(account, _plan(token, customer_id))
    _human_review(token, customer_id)
    _consent()


def _warning() -> None:
    st.warning(
        "**Your next payment may be at risk**\n\n"
        "Based on how your account has moved over the last six months, you may "
        "find your next payment difficult. Nothing has changed on your account, "
        "and this is not a decision — it is an early heads-up while there is "
        "still time to act."
    )


def _what_to_do(account: dict, plan: dict | None) -> None:
    # The computed plan says the same things with this account's own amounts in
    # them, so it replaces the written advice rather than repeating it.
    if plan and plan["steps"]:
        _recourse(plan)
    else:
        advice = _advice(account)
        if advice:
            st.markdown("**What would help most**")
            for because, action in advice:
                st.markdown(f"- {action}")
                st.caption(f"Because: {because.lower()}")

    st.markdown("**If money is tight this month**")
    st.markdown(
        "- Paying **any** amount is better than missing the payment entirely\n"
        "- Ask about a payment plan before the due date, not after\n"
        "- Contact support and the account can be put on hold while you arrange things"
    )


def _human_review(token: str, customer_id: str) -> None:
    """Ask the loan officer to look at the account. Offered to everyone, flagged or not."""
    st.markdown("**Talk to someone about your options**")
    try:
        open_request = api.open_review_request(customer_id, token)
    except api.ServiceUnavailable:
        st.caption("Requests for a human review are unavailable right now.")
        return

    if open_request:
        st.success(
            "**Your request has been sent to your loan officer**\n\n"
            f"Sent {review.format_time(open_request['requested_at'])}. "
            "Someone will look at your account and get back to you."
        )
        st.caption("Your comment")
        st.write(open_request["reason"])
        return

    comment = st.text_area(
        "Your comment for the loan officer",
        key=f"review_comment_{customer_id}",
        placeholder="Tell them what you would like them to look at.",
        max_chars=REVIEW_COMMENT_MAX,
    )
    if st.button(
        "Contact my loan officer for human review",
        key=f"review_send_{customer_id}", type="primary",
    ):
        if not comment.strip():
            st.error("Please leave a comment so the loan officer knows what to look at.")
            return
        try:
            api.request_review(customer_id, token, comment.strip())
        except api.ServiceUnavailable as exception:
            api.guard(exception)
            return
        st.rerun()


def _reassurance(account: dict) -> None:
    st.success(
        "**Your account looks on track**\n\n"
        "Nothing in your recent repayment history suggests you will have "
        "difficulty with your next payment."
    )

    helping = [
        reason["description"] for reason in account["reasons"]
        if not reason["raises_risk"]
    ]
    if helping:
        st.markdown("**What is working in your favour**")
        for description in helping[:SHOWN_ADVICE]:
            st.markdown(f"- {description}")

    st.caption(
        "This is a monthly check, so it can change as your account changes. "
        "You will be told if it does."
    )


def _account(account: dict) -> None:
    for column, (label, value) in zip(
        st.columns(4),
        [
            ("Credit limit", f"{account['credit_limit']:,.0f}"),
            ("Latest statement balance (BILL_AMT1)", f"{account['current_balance']:,.0f}"),
            ("Last payment", f"{account['last_payment']:,.0f}"),
            ("Months behind", f"{account['months_in_arrears']} of 6"),
        ],
    ):
        column.metric(label, value)


def _history(history: list[dict]) -> None:
    """Bill against payment for each month, with that month's repayment status above it."""
    if not history:
        return

    st.markdown("**Your last six months**")
    # The API sends most recent first; the chart reads oldest to newest.
    months = sorted(history, key=lambda month: month["month"], reverse=True)
    order = [month["label"] for month in months]

    status = pd.DataFrame(
        {
            "label": month["label"],
            "reading": month["payment_status_reading"].capitalize(),
            "behind": month["payment_status"] > 0,
        }
        for month in months
    )
    amounts = pd.DataFrame(
        {"label": month["label"], "series": series, "amount": month[field]}
        for month in months
        for series, field in (("Bill", "bill_amount"), ("Paid", "amount_paid"))
    )

    strip = (
        alt.Chart(status)
        .mark_text(fontSize=11, fontWeight=600, baseline="middle")
        .encode(
            x=alt.X("label:N", sort=order, title=None, axis=None),
            text="reading:N",
            color=alt.condition(
                alt.datum.behind, alt.value(theme.HIGH), alt.value(theme.MUTED)
            ),
        )
        .properties(height=22)
    )
    bars = (
        alt.Chart(amounts)
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            xOffset=alt.XOffset("series:N", sort=["Bill", "Paid"]),
            y=alt.Y("amount:Q", title=None, axis=alt.Axis(format="~s")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Bill", "Paid"], range=[theme.ACCENT, theme.MUTED]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Month"),
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("amount:Q", title="Amount", format=",.0f"),
            ],
        )
        .properties(height=220)
    )
    chart = alt.vconcat(strip, bars, spacing=4).resolve_scale(x="shared")
    st.altair_chart(chart, width="stretch", theme=None)


def _plan(token: str, customer_id: str) -> dict | None:
    """The computed recourse, or None if the search could not be reached.

    A failed search must not take the page down: the written advice below is
    always true and stands on its own.
    """
    try:
        return api.recourse(customer_id, token)
    except api.ServiceUnavailable:
        return None


def _recourse(plan: dict) -> None:
    """The way out, worked out against this account's own figures.

    Amounts rather than suggestions. When nothing the customer can do on their
    own reaches the next band, that is said: a recourse screen that always
    finds an answer is promising one.
    """
    st.markdown(f"**What would move you to {plan['target_band'].lower()} risk**")
    for step in plan["steps"]:
        st.markdown(f"- {step['description']}")

    if plan["reachable"]:
        st.caption(plan["note"])
    else:
        st.caption(
            f"{plan['note']} Each one still helps, and support can look at "
            "options this account cannot reach on its own."
        )


def _advice(account: dict) -> list[tuple[str, str]]:
    """Reasons the cardholder can act on, strongest first, paired with an action."""
    advice = []
    for reason in account["reasons"]:
        if not reason["raises_risk"]:
            continue
        action = ACTIONABLE.get(reason["feature"])
        if action:
            advice.append((reason["description"], action))
    return advice[:SHOWN_ADVICE]


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
            "messages": [{"role": "assistant", "content": GREETING, "note": "", "local": True}],
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

    asked = st.chat_input("Ask why, or what would help") or asked
    st.caption(conversation["disclaimer"])
    if not asked:
        return

    conversation["messages"].append({"role": "user", "content": asked, "note": ""})
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
                reply = {"role": "assistant", "content": turn["reply"], "note": turn["note"]}
            except api.ServiceUnavailable:
                reply = {
                    "role": "assistant",
                    "content": "I could not reach the service just now. Please try again.",
                    "note": "",
                }
    conversation["messages"].append(reply)
    st.rerun()


def _bubble(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("note"):
            st.caption(message["note"])


def _consent() -> None:
    with st.expander("How this check works, and your rights"):
        st.markdown(CONSENT)
        st.checkbox("Send me an early warning if my account looks at risk", value=True)
        st.caption("Preference not yet saved.")
