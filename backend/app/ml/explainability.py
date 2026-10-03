"""Local explanations: why did this customer get this score.

Uses LightGBM's native `pred_contrib`, which for a tree model is exact TreeSHAP
rather than sampled. Same quantity the `shap` package would compute, without the
large install.

Contributions are in log-odds and sum to the model's raw score, which the
calibrator then maps to the probability the user sees. The ordering and relative
size of the reasons are exact; only the final percentage comes from calibration.

Every user-facing string lives in `describe()`, so the UI formats numbers but never
invents wording.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

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
    "util_trend": "utilisation now versus six months ago",
    "repay_ratio_mean": "average share of the bill paid off",
    "repay_ratio_min": "worst month's share of the bill paid",
    "months_zero_payment": "months where nothing was paid",
    "bill_mean": "average bill over six months",
    "pay_amt_mean": "average payment over six months",
    "pay_to_bill_gap": "how far payments fall short of bills",
}


@dataclass
class Reason:
    """One driver of a single customer's score."""

    feature: str
    value: float
    effect: float  # log-odds, positive pushes risk up
    description: str

    @property
    def raises_risk(self) -> bool:
        return self.effect > 0

    def to_dict(self) -> dict:
        return asdict(self)


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


def describe(feature: str, value: float) -> str:
    """Readable account of what this feature's value says about the customer.

    Written for someone with no training: "No payment made last month", never
    "PAY_AMT1 = 0".
    """
    if feature == "LIMIT_BAL":
        return f"Credit limit of {value:,.0f}"
    if feature == "remaining_credit":
        if value <= 0:
            return "No credit left on the card"
        return f"{value:,.0f} of credit still available"

    if feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
        month = feature.split("_")[1]
        status = describe_pay_status(value)
        if month == "1":
            return f"Last month: {status}"
        return f"Month {month}: {status}"

    if feature == "pay_max":
        return f"Worst month in the six: {describe_pay_status(value)}"
    if feature == "pay_mean":
        return f"Average repayment status of {value:.1f} across six months"
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
        return f"Paid {value:,.0f} in month {month}"

    if feature.startswith("BILL_AMT"):
        return f"Bill of {value:,.0f} in month {feature[-1]}"

    if feature in ("util_mean", "util_max") or feature.startswith("util_"):
        if pd.isna(value):
            return f"{feature_meaning(feature).capitalize()} not available"
        return f"{feature_meaning(feature).capitalize()} at {value:.0%}"

    if feature.startswith("repay_ratio"):
        if pd.isna(value):
            return f"{feature_meaning(feature).capitalize()} not available"
        return f"{feature_meaning(feature).capitalize()} at {value:.0%}"

    if feature in ("bill_mean", "pay_amt_mean", "pay_to_bill_gap"):
        return f"{feature_meaning(feature).capitalize()}: {value:,.0f}"

    return f"{feature_meaning(feature).capitalize()}: {value:,.2f}"


class Explainer:
    """Per-customer reasons for a score.

    Holds the same ensemble `CreditScorer` serves, so the explanation describes the
    model that actually produced the number.
    """

    def __init__(self, scorer):
        self.scorer = scorer
        self.models = scorer.models

    def reasons(self, customer: dict, limit: int | None = None) -> tuple[list[Reason], float]:
        """Reasons for one customer, strongest first, with the base log-odds.

        The base value is the average customer's log-odds; the effects add to it to
        give this customer's raw score, so the explanation is complete.
        """
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
            raise ValueError(f"Expected exactly one row, got {len(features)}")

        # Shape (1, n_features + 1); the trailing column is the base value.
        stacked = np.mean(
            [model.predict_proba(features, pred_contrib=True) for model in self.models],
            axis=0,
        )
        row = np.asarray(stacked)[0]
        return row[:-1], float(row[-1])
