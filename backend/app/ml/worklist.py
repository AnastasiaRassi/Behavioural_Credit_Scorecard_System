"""Reading the officer worklist, and deciding what each role is shown.

The queue itself is precomputed by `train.py`. This module filters it, summarises
it, and reduces a row to what a cardholder may see.

That reduction happens here rather than in the frontend on purpose. If the API
returned a cardholder their probability and log-odds effects and the screen
merely declined to draw them, the numbers would still be in the response, one
devtools panel away. What a role cannot see, it is not sent.
"""
from app.ml import artifacts

BANDS = ("Low", "Medium", "High")

# A page of the worklist. The full queue is thousands of rows, which no officer
# scrolls and no response should carry.
DEFAULT_PAGE = 50
MAX_PAGE = 500


def all_rows() -> list[dict]:
    """Every queued customer, riskiest first."""
    rows = artifacts.officer_queue()
    return sorted(rows, key=lambda row: row["probability"], reverse=True)


def summarise(rows: list[dict]) -> dict:
    """Counts an officer needs before touching a filter."""
    bands = {band: 0 for band in BANDS}
    for row in rows:
        if row["risk_band"] in bands:
            bands[row["risk_band"]] += 1

    flagged = sum(1 for row in rows if row["flagged"])
    return {
        "in_queue": len(rows),
        "flagged": flagged,
        "not_flagged": len(rows) - flagged,
        "bands": bands,
    }


def filter_rows(
    rows: list[dict],
    *,
    band: str | None = None,
    flagged: bool | None = None,
    search: str | None = None,
    minimum_months_in_arrears: int | None = None,
) -> list[dict]:
    """Narrow the worklist. Unset filters do nothing."""
    selected = rows

    if band:
        selected = [row for row in selected if row["risk_band"] == band]
    if flagged is not None:
        selected = [row for row in selected if row["flagged"] is flagged]
    if minimum_months_in_arrears is not None:
        selected = [
            row for row in selected
            if row["months_in_arrears"] >= minimum_months_in_arrears
        ]
    if search:
        needle = search.strip().lower()
        selected = [
            row for row in selected if needle in row["customer_id"].lower()
        ]

    return selected


def find(customer_id: str) -> dict | None:
    return next(
        (row for row in all_rows() if row["customer_id"] == customer_id), None
    )


def highlights(row: dict) -> dict:
    """Facts an officer would otherwise work out by eye from the history.

    Derived here rather than in the UI (§1) and rather than stored, because every
    one of these follows from the six months the queue already carries; storing
    them would be a second copy able to disagree.
    """
    history = row.get("history") or []
    utilisations = [
        entry["utilisation"] for entry in history if entry["utilisation"] is not None
    ]
    statuses = [entry["payment_status"] for entry in history]

    return {
        "utilisation_now": utilisations[0] if utilisations else None,
        "utilisation_mean": (
            round(sum(utilisations) / len(utilisations), 4) if utilisations else None
        ),
        "trajectory": _trajectory(statuses),
        "months_since_full_payment": _months_since_full_payment(history),
        "months_with_no_payment": sum(
            1 for entry in history if entry["amount_paid"] <= 0
        ),
        "worst_payment_status": max(statuses) if statuses else None,
    }


def _trajectory(statuses: list[int]) -> str | None:
    """Whether arrears are deepening or easing across the six months.

    `statuses` is most-recent-first, so a lower first value than last means the
    account has improved. Ties are "steady" rather than forced either way.
    """
    if len(statuses) < 2:
        return None

    recent, oldest = statuses[0], statuses[-1]
    if recent > oldest:
        return "deteriorating"
    if recent < oldest:
        return "improving"
    return "steady"


def _months_since_full_payment(history: list[dict]) -> int | None:
    """Months back to the last time the bill was cleared.

    0 means the most recent month was paid in full. None means not in the six
    observed months, which is different from "a long time ago" and is left for
    the caller to word.
    """
    for entry in history:
        bill = entry["bill_amount"]
        if bill > 0 and entry["amount_paid"] >= bill:
            return entry["month"] - 1
        # A zero bill with nothing owed counts as settled.
        if bill <= 0:
            return entry["month"] - 1
    return None


def customer_detail(row: dict) -> dict:
    """Everything a dedicated customer page shows, for one queued customer."""
    return {
        "customer_id": row["customer_id"],
        "probability": row["probability"],
        "risk_band": row["risk_band"],
        "flagged": row["flagged"],
        "credit_limit": row["credit_limit"],
        "current_balance": row["current_balance"],
        "last_payment": row["last_payment"],
        "months_in_arrears": row["months_in_arrears"],
        "segment": row.get("segment"),
        "history": row.get("history") or [],
        "highlights": highlights(row),
        "reasons": row["reasons"],
    }


def cardholder_view(row: dict, stored_prediction=None) -> dict:
    """What a cardholder is told about their own account.

    Carries the band, whether the account needs attention, and the reasons in
    plain language. Deliberately omits the probability, the threshold and the
    log-odds effects: a cardholder is owed an explanation they can act on, not a
    number they cannot calibrate and would reasonably misread as a verdict.

    Reasons keep their direction, so the screen can separate what is driving the
    risk from what is working in the customer's favour.

    `stored_prediction`, when given, is a score an officer ran on this customer.
    Its band and reasons replace the queue's, so both sides of a conversation are
    looking at the same prediction. The reduction still applies: a fresher score
    does not mean a cardholder is sent the number.
    """
    band = row["risk_band"]
    needs_attention = row["flagged"]
    reasons = row["reasons"]

    if stored_prediction is not None:
        band = stored_prediction.risk_band
        needs_attention = stored_prediction.flagged
        stored_reasons = (stored_prediction.explanation or {}).get("reasons")
        if stored_reasons:
            reasons = stored_reasons

    return {
        "customer_id": row["customer_id"],
        "risk_band": band,
        "needs_attention": needs_attention,
        "credit_limit": row["credit_limit"],
        "current_balance": row["current_balance"],
        "last_payment": row["last_payment"],
        "months_in_arrears": row["months_in_arrears"],
        "reasons": [
            {
                "feature": reason["feature"],
                "description": reason["description"],
                # Stored reasons already carry the direction; queue rows carry
                # the effect it was derived from.
                "raises_risk": reason.get("raises_risk", reason.get("effect", 0) > 0),
            }
            for reason in reasons
        ],
    }
