"""Deterministic sanity checks on a score, so a model failure never ships a band that
contradicts the repayment history beside it.

Two tiers, both pure functions of the six payment statuses and the model's band:
  - floor: a Low band over a persistent or current delay is raised to Medium.
  - review: a Medium band over a deeply delinquent history stays, but is marked
    for an officer. Customers who recovered (PAY_1 back to 0) default near the
    Medium rate on held-out data, so forcing them to High would overshoot.
Malformed inputs are marked for review and never reshape the band.

The model's probability is never edited; the original band is kept as `model_band`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

BANDS = ("Low", "Medium", "High")
MONTHS = 6
PAY_MIN, PAY_MAX = -2, 9

# "Late" is a payment status of 1 or more months behind. Each rule fires on well under
# 1% of held-out customers, so this is a safety net, not an accuracy lever.
FLOOR_LATE_MONTHS = 3
FLOOR_LATEST_STATUS = 2
REVIEW_LATE_MONTHS = 4
REVIEW_SEVERE_MONTHS = 3  # months at 2+ behind

PERSISTENT_DELAY_FLOOR = "persistent_delay_floor"
CURRENT_DELAY_FLOOR = "current_delay_floor"
DEEP_DELAY_REVIEW = "deep_delay_review"
INVALID_INPUT = "invalid_input"

DESCRIPTIONS = {
    PERSISTENT_DELAY_FLOOR: f"{FLOOR_LATE_MONTHS}+ late months in the last {MONTHS}; band raised to Medium",
    CURRENT_DELAY_FLOOR: f"latest month {FLOOR_LATEST_STATUS}+ behind; band raised to Medium",
    DEEP_DELAY_REVIEW: "long delinquent history under a Medium band; officer review advised",
    INVALID_INPUT: "input outside the expected ranges; officer review advised",
}


@dataclass(frozen=True)
class GuardResult:
    band: str
    model_band: str
    flags: tuple[str, ...] = ()
    needs_review: bool = False

    @property
    def overridden(self) -> bool:
        return self.band != self.model_band


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def statuses_from_record(record: Mapping) -> list[float | None]:
    """PAY_1..PAY_6 (most recent first) from a raw customer mapping; None if unreadable."""
    return [_number(record.get(f"PAY_{i}")) for i in range(1, MONTHS + 1)]


def statuses_from_history(history: Sequence[Mapping]) -> list[float | None]:
    """The same list from a queue row's `history` (entries carry `month` 1..6)."""
    by_month = {entry["month"]: _number(entry.get("payment_status")) for entry in history}
    return [by_month.get(i) for i in range(1, MONTHS + 1)]


def input_problems(record: Mapping) -> list[str]:
    """Why a raw record is suspect; an empty list when it looks sane."""
    problems = []
    if any(s is None or not PAY_MIN <= s <= PAY_MAX for s in statuses_from_record(record)):
        problems.append("payment status missing or outside -2..9")
    limit = _number(record.get("LIMIT_BAL"))
    if limit is None or limit <= 0:
        problems.append("credit limit missing or not positive")
    for i in range(1, MONTHS + 1):
        if _number(record.get(f"BILL_AMT{i}")) is None:
            problems.append(f"BILL_AMT{i} missing")
        paid = _number(record.get(f"PAY_AMT{i}"))
        if paid is None or paid < 0:
            problems.append(f"PAY_AMT{i} missing or negative")
    return problems


def check(band: str, statuses: Sequence[float | None], *, valid_input: bool = True) -> GuardResult:
    """Apply the rules to `band` given PAY_1..PAY_6 (most recent first)."""
    if band not in BANDS:
        return GuardResult(band, band, (INVALID_INPUT,), True)
    if not valid_input or any(s is None for s in statuses):
        return GuardResult(band, band, (INVALID_INPUT,), True)

    late = sum(s >= 1 for s in statuses)
    severe = sum(s >= 2 for s in statuses)
    latest = statuses[0]

    if band == "Low":
        flags = []
        if late >= FLOOR_LATE_MONTHS:
            flags.append(PERSISTENT_DELAY_FLOOR)
        if latest >= FLOOR_LATEST_STATUS:
            flags.append(CURRENT_DELAY_FLOOR)
        if flags:
            return GuardResult("Medium", band, tuple(flags), True)
    elif band == "Medium":
        if late >= REVIEW_LATE_MONTHS or severe >= REVIEW_SEVERE_MONTHS or latest >= FLOOR_LATEST_STATUS:
            return GuardResult(band, band, (DEEP_DELAY_REVIEW,), True)
    return GuardResult(band, band)


def check_record(band: str, record: Mapping) -> GuardResult:
    """`check` for a raw customer mapping, including input sanity."""
    return check(band, statuses_from_record(record), valid_input=not input_problems(record))


def apply_to_row(row: dict) -> dict:
    """A queue row with the guard applied: guarded `risk_band`, plus the guard's fields.

    The row's `probability` is untouched. Rows already carrying `model_band` are
    returned as they are, so applying twice is harmless.
    """
    if "model_band" in row:
        return row
    result = check(row["risk_band"], statuses_from_history(row.get("history", [])))
    return {
        **row,
        "risk_band": result.band,
        "model_band": result.model_band,
        "guard_flags": list(result.flags),
        "needs_review": result.needs_review,
    }
