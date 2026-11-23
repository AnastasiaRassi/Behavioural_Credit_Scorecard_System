"""My Transactions: the cardholder's real repayments as a bank-style statement.

Each row is a month with a recorded payment (`history[].amount_paid`), newest first.
All six months show; the model predicts the next one. The dataset has no
account ids or channels, so those are invented but seeded by the customer id and
month, so they stay stable between reloads. Styles are in ui/transactions.css.
"""
import random
from html import escape

import streamlit as st

import api
from money import html_safe, to_usd
from ui import profile
from ui.components import flatten, show, stylesheet

CHANNELS = ("Online banking", "Mobile app", "Auto-debit", "ATM", "Branch")
CURRENCY_CODE = "USD"
COLUMNS = ("Month", "Reference", "Description", "Account ID", "Amount", "Direction", "Currency")


def card_for(seed: str) -> tuple[str, str]:
    """A fake card: its masked number and id, stable for one customer."""
    rng = random.Random(f"card:{seed}")
    return f"•••• •••• •••• {rng.randrange(10000):04d}", f"CC-{rng.randrange(10**8):08d}"


def account_for(seed: str) -> str:
    """A fake account id, stable for one customer."""
    return f"ACC-{random.Random(f'account:{seed}').randrange(10**8):08d}"


def rows_for(seed: str, history: list[dict]) -> list[dict]:
    """One row per paid month, newest first (month 1 is the most recent)."""
    account_id = account_for(seed)
    paid = sorted((month for month in history if month["amount_paid"] > 0),
                  key=lambda month: month["month"])
    rows = []
    for month in paid:
        rng = random.Random(f"tx:{seed}:{month['month']}")
        rows.append({
            "Month": month["label"],
            "Reference": f"TX{rng.randrange(10**9):09d}",
            "Description": f"Card repayment · {rng.choice(CHANNELS)}",
            "Account ID": account_id,
            "Amount": f"{to_usd(month['amount_paid']):,.2f}", "Direction": "Credit", "Currency": CURRENCY_CODE,
        })
    return rows


def render_page(session: dict, customer_id: str) -> None:
    try:
        history = api.cardholder(customer_id, session["token"]).get("history") or []
    except api.ServiceUnavailable as exception:
        api.guard(exception)
        return
    stylesheet("transactions.css")
    masked, card_id = card_for(customer_id)
    show(flatten(f"""
<div class="tx-card-face">
  <div class="tx-card-bank">{escape(profile.BANK_NAME)}</div>
  <div class="tx-card-number">{masked}</div>
  <div class="tx-card-foot"><span>{escape(session.get("display_name", "").upper())}</span><span>ID {card_id}</span></div>
</div>"""))
    rows = rows_for(customer_id, history)
    if not rows:
        st.info("No payments are recorded on this account yet.")
        return
    head = "".join(f"<th>{name}</th>" for name in COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{html_safe(escape(row[name]))}</td>" for name in COLUMNS) + "</tr>"
                   for row in rows)
    show(flatten(f'<table class="tx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'))
