"""The account's review record, drawn the same way on both screens.

A case, not a chat: each request is a full-width entry, and an officer's decision
is its own block with the outcome, name and date. Bodies are escaped, since they
are typed by people.
"""
from html import escape

import review
from ui.components import flatten, show

EMPTY = {
    "cardholder": "No requests yet. Submit one below.",
    "officer": "No requests or decisions on this account yet.",
}


def draw(entries: list[dict], viewer: str) -> None:
    """`viewer` is "cardholder" or "officer"; it only changes who a request is "from"."""
    if not entries:
        show(f'<div class="thread-empty">{EMPTY[viewer]}</div>')
        return

    cards = [_card(entry, viewer) for entry in entries]
    show(flatten(f'<div class="thread">{stepper(entries)}{"".join(cards)}</div>'))


def stepper(entries: list[dict]) -> str:
    """Submitted → Under review → Decision issued, for the newest request."""
    stage = review.case_stage(entries)
    if stage is None:
        return ""
    steps = []
    for number, label in enumerate(review.CASE_STEPS, start=1):
        state = "done" if number < stage else "current" if number == stage else "todo"
        steps.append(f'<li class="case-step case-step--{state}"><span>{number}</span>{label}</li>')
    return f'<ol class="case-steps">{"".join(steps)}</ol>'


def status_chip(entries: list[dict]) -> str:
    """The header badge: the case stage, grey when there is nothing left to do."""
    stage = review.case_stage(entries)
    if stage is None:
        return '<span class="chat-status chat-status--closed">No open request</span>'
    tone = "closed" if review.is_closed(entries) else "active"
    return f'<span class="chat-status chat-status--{tone}">{review.CASE_STEPS[stage - 1]}</span>'


def _card(entry: dict, viewer: str) -> str:
    when = escape(review.format_time(entry["at"]))
    author = escape(entry["author"])

    if entry["kind"] == "decision":
        band = entry.get("band")
        outcome = escape(
            f"Risk level set to {band}" if band else review.ACTIONS.get(entry["action"], entry["action"])
        )
        return (
            '<div class="case-entry case-entry--decision">'
            '<div class="case-decision-head"><span class="case-label">Decision issued</span>'
            f'<span class="case-outcome">{outcome}</span></div>'
            f'<div class="case-body">{escape(entry["body"])}</div>'
            f'<div class="case-sign">{author} · Loan officer · {when}</div></div>'
        )

    label = "Your request" if viewer == "cardholder" else "Customer request"
    awaiting = entry["status"] in review.OPEN_STATUSES
    tag = escape(review.REQUEST_TAGS.get(entry["status"], entry["status"]))
    kind = "open" if awaiting else "done"
    kind = "done" if entry["status"] == "rejected" else kind
    return (
        f'<div class="case-entry case-entry--request case-entry--{kind}">'
        f'<div class="case-meta"><span class="case-label">{label}</span>'
        f'<span class="case-tag case-tag--{kind}">{tag}</span></div>'
        f'<div class="case-body">{escape(entry["body"])}</div>'
        f'<div class="case-sign">{author} · {when}</div></div>'
    )
