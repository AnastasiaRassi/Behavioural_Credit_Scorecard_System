"""Pure logic behind the officer's review screen: no Streamlit, so it is unit-tested.

A review request is a customer asking a person to look again at the model's rating.
The shapes here are the ones `api.review_requests()` returns.
"""
from datetime import datetime, timezone

OPEN_STATUSES = ("pending", "in_review")
STATUS_LABELS = {"pending": "Pending", "in_review": "In review", "resolved": "Resolved"}

NOTE_PREVIEW_CHARS = 140


def parse_time(value: str) -> datetime:
    """An ISO timestamp as an aware UTC datetime; a naive one is taken as UTC."""
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def waiting_text(requested_at: str, now: datetime) -> str:
    """How long a request has waited, in the largest whole unit."""
    seconds = max((now - parse_time(requested_at)).total_seconds(), 0)
    minutes = int(seconds // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return _plural(minutes, "minute")
    if minutes < 24 * 60:
        return _plural(minutes // 60, "hour")
    return _plural(minutes // (24 * 60), "day")


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}{'' if count == 1 else 's'}"


def truncate(text: str, limit: int = NOTE_PREVIEW_CHARS) -> str:
    """The text cut at a word boundary, with an ellipsis when something was left out."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def open_requests(requests: list[dict]) -> list[dict]:
    """Requests still needing a person, oldest first."""
    waiting = [item for item in requests if item["status"] in OPEN_STATUSES]
    return sorted(waiting, key=lambda item: parse_time(item["requested_at"]))


def pending_count(requests: list[dict]) -> int:
    return sum(item["status"] == "pending" for item in requests)


def request_for(requests: list[dict], customer_id: str | None) -> dict | None:
    """The oldest open request this customer has made, if any."""
    for item in open_requests(requests):
        if item["customer_id"] == customer_id:
            return item
    return None


def card_data(request: dict, band: str | None, now: datetime) -> dict:
    """Everything one request card draws, with the wording settled."""
    note = request["reason"].strip()
    preview = truncate(note)
    return {
        "name": request.get("full_name") or f"Customer {request['account_number']}",
        "account": request["account_number"],
        "waiting": waiting_text(request["requested_at"], now),
        "note": preview,
        "full_note": note,
        "is_truncated": preview != " ".join(note.split()),
        "status": STATUS_LABELS.get(request["status"], request["status"]),
        "band": band,
    }


def model_saw(detail: dict) -> list[str]:
    """The last six months in plain sentences, from the history the screen already has."""
    highlights = detail["highlights"]
    sentences = []

    behind = detail["months_in_arrears"]
    sentences.append(
        "Never behind on a payment in the last six months."
        if behind == 0
        else f"Behind on payments in {behind} of the last 6 months."
    )

    missed = highlights.get("months_with_no_payment")
    if missed:
        sentences.append(f"No payment at all in {_plural(missed, 'month')}.")

    now, average = highlights.get("utilisation_now"), highlights.get("utilisation_mean")
    if now is not None and average is not None:
        sentences.append(
            f"Using {now:.0%} of the limit now, {average:.0%} on average over six months."
        )

    billed = sum(month["bill_amount"] for month in detail["history"])
    paid = sum(month["amount_paid"] for month in detail["history"])
    if billed > 0:
        sentences.append(
            f"Paid {paid:,.0f} against {billed:,.0f} billed ({paid / billed:.0%})."
        )

    trajectory = highlights.get("trajectory")
    if trajectory:
        sentences.append(f"Repayment behaviour is {trajectory}.")
    return sentences


def unchangeable_reason(reasons: list[dict], actionable: set[str]) -> str | None:
    """The top reason raising the risk, if the customer cannot act on it."""
    for reason in reasons:
        if reason.get("effect", 0) > 0:
            return None if reason["feature"] in actionable else reason["description"]
    return None


def format_time(value: str) -> str:
    return parse_time(value).strftime("%d %b %Y, %H:%M")
