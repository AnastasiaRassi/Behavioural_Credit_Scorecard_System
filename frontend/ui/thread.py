"""The account's activity thread, drawn the same way on both screens.

Cardholder messages and officer decisions read like a chat: the viewer's own
side sits on the right. Bodies are escaped, since they are typed by people.
"""
from html import escape

import review
from ui.components import flatten, show

EMPTY = {
    "cardholder": "No messages yet. Write to your loan officer below.",
    "officer": "No messages or decisions on this account yet.",
}


def draw(entries: list[dict], viewer: str) -> None:
    """`viewer` is "cardholder" or "officer"; it decides which side is "mine"."""
    if not entries:
        show(f'<div class="thread-empty">{EMPTY[viewer]}</div>')
        return

    cards = [_card(entry, viewer) for entry in entries]
    show(flatten(f'<div class="thread">{"".join(cards)}</div>'))


def _card(entry: dict, viewer: str) -> str:
    is_decision = entry["kind"] == "decision"
    mine = is_decision == (viewer == "officer")
    side = "mine" if mine else "theirs"
    when = review.format_time(entry["at"])

    if is_decision:
        action = review.ACTIONS.get(entry["action"], entry["action"])
        who = "You" if mine else f"{entry['author']} · Loan officer"
        head = (
            f'<span class="thread-tag thread-tag--decision">Decision · {escape(action)}</span>'
        )
        foot = "Conversation closed"
        kind = "decision"
    else:
        who = "You" if mine else entry["author"]
        awaiting = entry["status"] in review.OPEN_STATUSES
        head = (
            '<span class="thread-tag thread-tag--open">Awaiting reply</span>'
            if awaiting else ""
        )
        foot = ""
        kind = "message--open" if awaiting else "message"

    footer = f'<div class="thread-foot">{foot}</div>' if foot else ""
    return (
        f'<div class="thread-row thread-row--{side}">'
        f'<div class="thread-card thread-card--{kind}">'
        f'<div class="thread-meta"><b>{escape(who)}</b> · {escape(when)}</div>'
        f"{head}"
        f'<div class="thread-body">{escape(entry["body"])}</div>'
        f"{footer}</div></div>"
    )
