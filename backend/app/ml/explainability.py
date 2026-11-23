"""Local explanations: why did this customer get this score.

Uses LightGBM's native `pred_contrib`, which for a tree model is exact TreeSHAP
rather than sampled. Same quantity the `shap` package would compute, without the
large install.

Contributions are in log-odds and average over the seed ensemble. Their ordering
and relative size are exact; the probability shown comes from the ensemble's
averaged output (see `calibration.py`), not from summing the contributions.

Every user-facing string lives in `describe()`, so the UI formats numbers but never
invents wording.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app.core.money import money
from app.ml.exceptions import ExplanationFailed
from app.ml.feature_engineering import describe_pay_status

# Reasons below this absolute log-odds effect are noise next to the top drivers.
NEGLIGIBLE_EFFECT = 0.005

# Plain meanings for engineered features, used by the global importance table.
FEATURE_MEANINGS = {
    "LIMIT_BAL": "the customer's credit limit",
    "pay_max": "worst delinquency in the six months",
    "pay_mean": "average repayment status",
    "pay_months_late": "months in arrears out of six",
    "pay_months_revolving": "months carrying a balance without arrears",
    "pay_trend": "delinquency now versus six months ago",
    "pay_worsening": "how many months got worse",
    "remaining_credit": "limit minus current balance",
    "util_mean": "average share of the limit used",
    "util_max": "highest share of the limit used",
    "util_trend": "share of the limit used now versus six months ago",
    "repay_ratio_mean": "average share of the bill paid off",
    "repay_ratio_min": "worst month's share of the bill paid",
    "months_zero_payment": "months where nothing was paid",
    "bill_mean": "average bill over six months",
    "pay_amt_mean": "average payment over six months",
    "pay_to_bill_gap": "how far payments fall short of bills",
}


@dataclass
class Reason:
    """One driver of a single customer's score: a feature, or a concept grouping several."""

    feature: str
    value: float | None
    effect: float  # log-odds, positive pushes risk up
    description: str
    # Concept reasons only: the features summed into `effect`, strongest first.
    parts: list[dict] | None = None

    @property
    def raises_risk(self) -> bool:
        return self.effect > 0

    def to_dict(self) -> dict:
        data = asdict(self)
        # NaN (a feature with nothing to compute from) is not valid JSON: Postgres
        # rejects it and it breaks the response. The description already says so.
        if data["value"] is not None and pd.isna(data["value"]):
            data["value"] = None
        if data["parts"] is None:
            del data["parts"]
        return data


# The dataset's monthly columns are positional, not dated: PAY_1 is the most
# recent observed month and PAY_6 the oldest. Labels live here rather than in the
# UI (§12), so the queue, the API and any screen name the same month identically.
MONTH_LABELS = {
    1: "Last month",
    2: "2 months ago",
    3: "3 months ago",
    4: "4 months ago",
    5: "5 months ago",
    6: "6 months ago",
}

MONTHS_OBSERVED = len(MONTH_LABELS)

# pay_mean is the average of six PAY_x codes (-2 no balance, -1 paid in full, 0
# revolving, 1+ months in arrears). Printing it ("4.7") reads as a measurement
# nobody can interpret, so it is banded on that scale: a mean of 1 or more is
# on average at least a month behind, a mean at or below 0 is on average not
# behind at all, and anything between is the occasional late month.
PAY_MEAN_USUALLY_BEHIND = 1.0
PAY_MEAN_USUALLY_CURRENT = 0.0


def month_label(position: int) -> str:
    """Readable name for a monthly column's position, 1 being the most recent."""
    return MONTH_LABELS.get(position, f"month {position}")


def feature_meaning(feature: str) -> str:
    """Plain-language gloss for a feature name, covering the raw monthly columns."""
    if feature in FEATURE_MEANINGS:
        return FEATURE_MEANINGS[feature]
    if feature.startswith("PAY_AMT"):
        return f"amount paid, month {feature[-1]}"
    if feature.startswith("BILL_AMT"):
        return f"bill amount, month {feature[-1]}"
    if feature.startswith("PAY_"):
        return f"repayment status, month {feature[-1]}"
    if feature.startswith("util_"):
        return f"share of limit used, month {feature[-1]}"
    if feature.startswith("repay_ratio_"):
        return f"share of bill paid, month {feature[-1]}"
    return feature.replace("_", " ")


def describe_change(feature: str, before: float, after: float) -> str:
    """What a customer would have to do, in the words they would do it in.

    `describe()` says what a number is; this says what to change it to. Only the
    features counterfactuals.default_policy lets a customer act on reach here —
    this month's bill, this month's payment, this month's repayment status, and
    the limit when a lender is modelling an increase.
    """
    moved = after - before

    if feature == "BILL_AMT1":
        return f"Bring the card balance down to {money(after)}, from {money(before)}"

    if feature == "PAY_AMT1":
        if before == 0:
            return f"Put {money(after)} into the card this month, instead of nothing"
        return f"Pay {money(after)} into the card this month, {money(moved, signed=True)} on now"

    if feature == "PAY_1":
        if after <= 0 < before:
            return "Bring the account up to date, so no month is in arrears"
        months = int(round(before - after))
        return f"Clear {months} of the months the account is behind"

    if feature == "LIMIT_BAL":
        return f"Raise the credit limit to {money(after)}, from {money(before)}"

    # Nothing else is actionable today, so this is a safety net rather than a
    # branch anyone should hit: say the move plainly instead of guessing at it.
    return f"{feature_meaning(feature).capitalize()}: {before:,.0f} to {after:,.0f}"


def describe(feature: str, value: float) -> str:
    """Readable account of what this feature's value says about the customer.

    Written for someone with no training: "No payment made last month", never
    "PAY_AMT1 = 0".
    """
    if feature == "LIMIT_BAL":
        return f"Credit limit of {money(value)}"
    if feature == "remaining_credit":
        if value <= 0:
            return "No credit left on the card"
        return f"{money(value)} of credit still available"

    if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
        month = feature.split("_")[1]
        status = describe_pay_status(value)
        if month == "1":
            return f"Last month: {status}"
        return f"Month {month}: {status}"

    if feature == "pay_max":
        # Status <= 0 means never in arrears; "no balance" reads oddly as a "worst month".
        if value <= 0:
            return "Never behind on payments in the last six months"
        return f"Worst month in the six: {describe_pay_status(value)}"
    if feature == "pay_mean":
        if value >= PAY_MEAN_USUALLY_BEHIND:
            return "Usually behind on repayments across the six months"
        if value > PAY_MEAN_USUALLY_CURRENT:
            return "Sometimes behind on repayments across the six months"
        return "Usually up to date on repayments across the six months"
    if feature == "pay_months_late":
        if value == 0:
            return "Never in arrears in the last six months"
        return f"In arrears {int(value)} of the last six months"
    if feature == "pay_months_revolving":
        return f"Carried a balance without arrears for {int(value)} months"
    if feature == "pay_trend":
        if value > 0:
            return "Repayment status has worsened since six months ago"
        if value < 0:
            return "Repayment status has improved since six months ago"
        return "Repayment status unchanged over six months"
    if feature == "pay_worsening":
        return f"Repayment status got worse in {int(value)} of the months"

    if feature == "months_zero_payment":
        if value == 0:
            return "Made a payment every month"
        if value == 1:
            return "One month with no payment made"
        return f"{int(value)} months with no payment made"

    if feature.startswith("PAY_AMT"):
        month = feature[-1]
        if value == 0:
            return f"No payment made in month {month}"
        return f"Paid {money(value)} in month {month}"

    if feature.startswith("BILL_AMT"):
        return f"Bill of {money(value)} in month {feature[-1]}"

    # A difference of two shares, so "at 46%" would misstate it: say which way it moved.
    if feature == "util_trend":
        if pd.isna(value):
            return f"{feature_meaning(feature).capitalize()} not available"
        points = round(abs(value) * 100)
        if points == 0:
            return "Share of the limit in use is unchanged since six months ago"
        direction = "up" if value > 0 else "down"
        return f"Share of the limit in use is {direction} {points} percentage points since six months ago"

    if feature in ("util_mean", "util_max") or feature.startswith("util_"):
        if pd.isna(value):
            return f"{feature_meaning(feature).capitalize()} not available"
        return f"{feature_meaning(feature).capitalize()} at {value:.0%}"

    if feature.startswith("repay_ratio"):
        if pd.isna(value):
            return f"{feature_meaning(feature).capitalize()} not available"
        return f"{feature_meaning(feature).capitalize()} at {value:.0%}"

    # Mean bill minus mean payment, so the sign matters and a bare figure hides it.
    if feature == "pay_to_bill_gap":
        if value > 0:
            return f"Payments fall short of the average bill by {money(value)} a month"
        if value < 0:
            return f"Payments exceed the average bill by {money(-value)} a month"
        return "Payments match the average bill"

    if feature in ("bill_mean", "pay_amt_mean"):
        return f"{feature_meaning(feature).capitalize()}: {money(value)}"

    return f"{feature_meaning(feature).capitalize()}: {value:,.2f}"


# ---- concepts --------------------------------------------------------------
#
# Many served features measure the same thing (BILL_AMT1, util_1, remaining_credit),
# so per-feature SHAP splits credit between them arbitrarily and signs flip: a 3%
# repayment can show as lowering risk because a correlated feature absorbed the
# rest. SHAP is additive, so summing within a concept is exact, and the sentence is
# written from the customer's own figures rather than stitched from feature labels.

CONCEPTS = {
    "repayment_status": "Repayment status",
    "share_paid": "Share of the bill paid",
    "carrying_balance": "Carrying a balance",
    "credit_in_use": "Credit in use",
    "account_size": "Account size and activity",
    "recent_direction": "Recent direction",
    "credit_limit": "Credit limit",
}

_CONCEPT_OF = {
    **{f"PAY_{i}": "repayment_status" for i in range(1, 7)},
    "pay_max": "repayment_status",
    "pay_mean": "repayment_status",
    "pay_months_late": "repayment_status",
    **{f"repay_ratio_{i}": "share_paid" for i in range(1, 6)},
    "repay_ratio_mean": "share_paid",
    "repay_ratio_min": "share_paid",
    "months_zero_payment": "share_paid",
    "pay_months_revolving": "carrying_balance",
    "pay_to_bill_gap": "carrying_balance",
    **{f"util_{i}": "credit_in_use" for i in range(1, 7)},
    "util_mean": "credit_in_use",
    "util_max": "credit_in_use",
    "remaining_credit": "credit_in_use",
    # Absolute amounts measure scale; how much of the bill is paid lives in the ratios.
    **{f"BILL_AMT{i}": "account_size" for i in range(1, 7)},
    **{f"PAY_AMT{i}": "account_size" for i in range(1, 7)},
    "bill_mean": "account_size",
    "pay_amt_mean": "account_size",
    "pay_trend": "recent_direction",
    "pay_worsening": "recent_direction",
    "util_trend": "recent_direction",
    "LIMIT_BAL": "credit_limit",
}

# Higher than NEGLIGIBLE_EFFECT: a concept this weak is not worth a line on screen.
NEGLIGIBLE_CONCEPT = 0.02
CLEARED_SHARE = 0.99  # a payment within 1% of the bill counts as clearing it
AT_LIMIT_SHARE = 0.95
INACTIVE_MONTHS = 2  # a balance in this many of six months or fewer reads as inactive


def concept_of(feature: str) -> str:
    try:
        return _CONCEPT_OF[feature]
    except KeyError:
        raise ExplanationFailed(
            f"Feature {feature!r} has no concept; add it to _CONCEPT_OF",
            context={"feature": feature},
        ) from None


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _join(parts: list[str]) -> str:
    return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]


def _repayment_status(row: pd.Series, raises: bool) -> str:
    status = [int(row[f"PAY_{i}"]) for i in range(1, 7)]
    behind = sum(s > 0 for s in status)
    if behind:
        worst = max(status)
        now = ("including last month" if status[0] > 0
               else "but up to date last month")
        return (f"behind in {behind} of the last 6 months, at worst "
                f"{_count(worst, 'month')} late, {now}")
    # Not behind: the codes still separate paid in full, carried over and no balance.
    mix = [(sum(s == -1 for s in status), "paid in full"),
           (sum(s == 0 for s in status), "carried part of the bill over"),
           (sum(s == -2 for s in status), "had nothing to pay")]
    detail = _join([f"{text} in {_count(n, 'month')}" for n, text in mix if n])
    return f"never behind in the last 6 months; {detail}"


def _share_paid(row: pd.Series, raises: bool) -> str:
    ratios = [row[f"repay_ratio_{i}"] for i in range(1, 6)]
    owed = [r for r in ratios if pd.notna(r)]
    if not owed:
        return "no bills to pay in the last 5 months"
    cleared = sum(r >= CLEARED_SHARE for r in owed)
    unpaid = sum(r == 0 for r in owed)
    if len(owed) == 1:
        text = "cleared the one bill due" if cleared else "did not clear the one bill due"
    else:
        text = f"cleared the bill in full in {cleared} of the {len(owed)} months a bill was due"
    if pd.notna(ratios[0]):
        latest = "all" if ratios[0] >= CLEARED_SHARE else f"{ratios[0]:.0%}"
        text += f"; the latest payment covered {latest} of the bill it was for"
    if unpaid:
        text += f"; nothing was paid on {_count(unpaid, 'bill')}"
    return text


def _carrying_balance(row: pd.Series, raises: bool) -> str:
    months = int(row["pay_months_revolving"])
    text = ("never carried part of a bill over" if months == 0
            else f"carried part of the bill over in {months} of the last 6 months")
    gap = row["pay_to_bill_gap"]
    if gap > 0:
        text += f"; payments averaged {money(gap)} a month less than the bills"
    return text


def _credit_in_use(row: pd.Series, raises: bool) -> str:
    used, remaining, peak = row["util_1"], row["remaining_credit"], row["util_max"]
    if pd.isna(used):
        return "share of the limit in use is not available"
    if remaining < 0:
        return f"over the limit by {money(-remaining)}, using {used:.0%} of it"
    if used >= AT_LIMIT_SHARE:
        return f"at the limit, using {used:.0%} of it"
    text = f"using {used:.0%} of the limit, with {money(remaining)} still free"
    if pd.notna(peak) and peak - used >= 0.10:
        text += f"; it peaked at {peak:.0%} in the last 6 months"
    return text


def _account_size(row: pd.Series, raises: bool) -> str:
    # Payment amounts are left out on purpose: next to the bill they read as
    # "you paid too little", which is the share-paid concept, not this one.
    active = sum(row[f"BILL_AMT{i}"] > 0 for i in range(1, 7))
    if active == 0:
        return "no balance on the card in the last 6 months"
    text = (f"bills averaged {money(row['bill_mean'])} a month, "
            f"with a balance in {active} of the last 6 months")
    if active <= INACTIVE_MONTHS:
        text = f"a small, mostly inactive account: {text}"
    why = ("smaller, less active accounts miss payments more often" if raises
           else "larger, active accounts miss payments less often")
    return f"{text} ({why})"


def _recent_direction(row: pd.Series, raises: bool) -> str:
    # Compared on months behind, not raw codes: a code moving -2 to 0 is a new
    # balance, not a worse repayment.
    now, then = max(int(row["PAY_1"]), 0), max(int(row["PAY_6"]), 0)
    if now > then:
        status = "further behind than 6 months ago"
    elif now < then:
        status = "less behind than 6 months ago"
    elif now == 0:
        status = "not behind now or 6 months ago"
    else:
        status = "as far behind as 6 months ago"
    trend = row["util_trend"]
    if pd.isna(trend) or round(abs(trend) * 100) == 0:
        return f"{status}; share of the limit in use unchanged"
    direction = "up" if trend > 0 else "down"
    return f"{status}; share of the limit in use {direction} {round(abs(trend) * 100)} points"


def _credit_limit(row: pd.Series, raises: bool) -> str:
    why = ("lower limits go with more missed payments" if raises
           else "higher limits go with fewer missed payments")
    return f"a limit of {money(row['LIMIT_BAL'])} ({why})"


_SENTENCES = {
    "repayment_status": _repayment_status,
    "share_paid": _share_paid,
    "carrying_balance": _carrying_balance,
    "credit_in_use": _credit_in_use,
    "account_size": _account_size,
    "recent_direction": _recent_direction,
    "credit_limit": _credit_limit,
}


def describe_concept(concept: str, row: pd.Series, effect: float) -> str:
    """One sentence for a concept, from the customer's own figures."""
    return f"{CONCEPTS[concept]}: {_SENTENCES[concept](row, effect > 0)}"


def concept_reasons(row: pd.Series, effects, columns, limit: int | None = None) -> list[Reason]:
    """Group one customer's per-feature contributions into concept reasons, strongest first."""
    totals: dict[str, float] = {key: 0.0 for key in CONCEPTS}
    parts: dict[str, list[dict]] = {key: [] for key in CONCEPTS}
    for name, effect in zip(columns, effects):
        concept = concept_of(name)
        totals[concept] += float(effect)
        value = float(row[name])
        parts[concept].append({
            "feature": name,
            "value": None if pd.isna(value) else value,
            "effect": float(effect),
            "description": describe(name, value),
        })

    ranked = sorted(
        (
            Reason(
                feature=key,
                value=None,
                effect=total,
                description=describe_concept(key, row, total),
                parts=sorted(parts[key], key=lambda p: abs(p["effect"]), reverse=True),
            )
            for key, total in totals.items()
            if abs(total) >= NEGLIGIBLE_CONCEPT
        ),
        key=lambda reason: abs(reason.effect),
        reverse=True,
    )
    return ranked[:limit] if limit else ranked


class Explainer:
    """Per-customer reasons for a score.

    Holds the same ensemble `CreditScorer` serves, so the explanation describes the
    model that actually produced the number.
    """

    def __init__(self, scorer):
        self.scorer = scorer
        self.models = scorer.models

    def reasons(self, customer: dict, limit: int | None = None) -> tuple[list[Reason], float]:
        """Concept reasons for one customer, strongest first, with the base log-odds.

        The base value is the average customer's log-odds; the effects add to it to
        give this customer's raw score, so the explanation is complete.
        """
        features = self.scorer.prepare(pd.DataFrame([customer]))
        effects, base = self._contributions(features)
        return concept_reasons(features.iloc[0], effects, features.columns, limit), base

    def feature_reasons(self, customer: dict, limit: int | None = None) -> tuple[list[Reason], float]:
        """Per-feature reasons, for the ML engineer's view. Signs between correlated
        features are not meaningful on their own; customers see `reasons()`."""
        features = self.scorer.prepare(pd.DataFrame([customer]))
        effects, base = self._contributions(features)

        ranked = sorted(
            (
                Reason(
                    feature=name,
                    value=float(features.iloc[0][name]),
                    effect=float(effect),
                    description=describe(name, float(features.iloc[0][name])),
                )
                for name, effect in zip(features.columns, effects)
                if abs(effect) >= NEGLIGIBLE_EFFECT
            ),
            key=lambda reason: abs(reason.effect),
            reverse=True,
        )
        return (ranked[:limit] if limit else ranked), base

    def _contributions(self, features: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Mean per-feature log-odds contributions across the seed ensemble."""
        if len(features) != 1:
            raise ExplanationFailed(
                f"Expected exactly one row, got {len(features)}", context={"rows": len(features)}
            )

        # Shape (1, n_features + 1); the trailing column is the base value.
        stacked = np.mean(
            [model.predict_proba(features, pred_contrib=True) for model in self.models],
            axis=0,
        )
        row = np.asarray(stacked)[0]
        return row[:-1], float(row[-1])
