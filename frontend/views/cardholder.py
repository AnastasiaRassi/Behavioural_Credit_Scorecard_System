"""Cardholder view: an early warning the customer can act on.

Shows no probability and no threshold, because the API does not send them. That
enforcement is the backend's (`worklist.cardholder_view`), not this screen's: a
page that merely declined to draw the number would still have received it.

Telling someone they are predicted to default is distressing, can be
self-fulfilling, and commits the lender to specific reasons the moment it acts.
Pre-delinquency contact is framed as assistance, so this screen says what to do.
"""
import streamlit as st

import api

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

CONSENT = """
Your account is checked each month using your billing and repayment history for
the last six months.

The check never uses your sex, age, education or marital status. Those are
excluded from the model entirely, and are kept only so the bank can test the
system for fairness.

No decision about your account is made automatically. A member of staff reviews
anything that needs action, and you can ask for a human review of any decision.
"""


def render() -> None:
    account = _signed_in_as()
    if account is None:
        return

    st.divider()
    if account["needs_attention"]:
        _warning(account)
    else:
        _reassurance(account)

    st.divider()
    _account(account)
    _consent()


def _signed_in_as() -> dict | None:
    """Stands in for a sign-in, which does not exist yet."""
    st.caption(
        "Signed in as — a real cardholder would see only their own account. "
        "This picker stands in for authentication, which is not built yet."
    )

    try:
        known = api.users(role="cardholder")
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return None

    if not known:
        st.info("No cardholder accounts seeded. Run: `cd backend && python -m app.db.seed`")
        return None

    chosen = st.selectbox(
        "Account",
        [user["display_name"] for user in known],
        format_func=lambda name: f"Account {name}",
    )
    try:
        return api.cardholder(chosen)
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return None


def _warning(account: dict) -> None:
    st.warning(
        "**Your next payment may be at risk**\n\n"
        "Based on how your account has moved over the last six months, you may "
        "find your next payment difficult. Nothing has changed on your account, "
        "and this is not a decision — it is an early heads-up while there is "
        "still time to act."
    )

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
    st.button("Talk to someone about my options", type="primary")
    st.caption("Not yet connected to a support queue.")


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
            ("Current balance", f"{account['current_balance']:,.0f}"),
            ("Last payment", f"{account['last_payment']:,.0f}"),
            ("Months behind", f"{account['months_in_arrears']} of 6"),
        ],
    ):
        column.metric(label, value)


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


def _consent() -> None:
    with st.expander("How this check works, and your rights"):
        st.markdown(CONSENT)
        st.checkbox("Send me an early warning if my account looks at risk", value=True)
        st.caption("Preference not yet saved.")
