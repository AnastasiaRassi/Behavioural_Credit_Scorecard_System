"""Smoke checks for counterfactual recourse.

The two properties that matter are that the advice is *valid* — re-scoring the
changed customer really does land in the target band — and that it is *legal*: no
immutable feature moves, and nothing moves in a direction the customer cannot act in.

Run from the backend/ directory:

    python -m pytest tests/test_counterfactuals.py
"""
import pytest

from app.ml.counterfactuals import (
    ArrearsConsistency,
    CounterfactualExplainer,
    GreedySearch,
    NoConsistency,
    RecoursePolicy,
    RecourseTarget,
    RuleBasedSearch,
    default_policy,
)
from app.ml.feature_engineering import PROTECTED
from app.ml.scorer import CreditScorer, example_customer

# A customer the model puts in the High band: two months in arrears, nothing paid,
# card almost at the limit.
STRESSED = example_customer() | {
    "PAY_1": 2, "PAY_2": 2, "PAY_3": 1,
    "BILL_AMT1": 195_000, "PAY_AMT1": 0, "PAY_AMT2": 0,
}

IMMUTABLE = [*PROTECTED, "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT3", "PAY_AMT4"]


@pytest.fixture(scope="module")
def scorer() -> CreditScorer:
    return CreditScorer()


@pytest.fixture(scope="module")
def explainer(scorer: CreditScorer) -> CounterfactualExplainer:
    return CounterfactualExplainer(scorer)


@pytest.fixture(scope="module")
def recourse(explainer: CounterfactualExplainer):
    return explainer.explain(STRESSED, RecourseTarget(band="Low"))


# --- validity -------------------------------------------------------------


def test_a_high_risk_customer_has_recourse(recourse):
    assert recourse.is_feasible
    assert recourse.feature_changes
    assert recourse.original_band == "High"


def test_the_counterfactual_really_scores_in_the_target_band(recourse, scorer: CreditScorer):
    """Re-score the changed customer end to end; the reported score must hold up."""
    changed = STRESSED | {name: c["to"] for name, c in recourse.feature_changes.items()}
    assert scorer.score_one(changed).risk_band == "Low"
    assert abs(scorer.score_one(changed).probability - recourse.achieved_score) < 0.01


def test_the_achieved_score_clears_the_ceiling(recourse):
    assert recourse.achieved_score <= recourse.target_score
    assert recourse.achieved_score < recourse.original_score


def test_deltas_match_the_endpoints(recourse):
    for change in recourse.feature_changes.values():
        assert abs((change["to"] - change["from"]) - change["delta"]) < 1e-9


# --- constraints ----------------------------------------------------------


def test_immutable_features_never_move(recourse):
    assert not set(recourse.feature_changes) & set(IMMUTABLE)


def test_changes_respect_the_allowed_direction(recourse):
    bounds = default_policy(STRESSED).bounds
    for name, change in recourse.feature_changes.items():
        bound = bounds[name]
        assert bound.minimum - 1e-6 <= change["to"] <= bound.maximum + 1e-6
        if name in ("BILL_AMT1", "PAY_1"):
            assert change["delta"] < 0, f"{name} should only fall"
        if name == "PAY_AMT1":
            assert change["delta"] > 0, "a customer cannot un-pay a bill"


def test_the_credit_limit_is_the_lenders_lever_not_the_customers(recourse):
    assert "LIMIT_BAL" not in recourse.feature_changes
    assert "LIMIT_BAL" in default_policy(STRESSED, allow_limit_increase=True).bounds


def test_clearing_arrears_requires_paying_for_it(recourse):
    """A better repayment status with no payment behind it is not recourse."""
    if "PAY_1" in recourse.feature_changes:
        paid = recourse.feature_changes["PAY_AMT1"]["to"]
        assert paid >= STRESSED["BILL_AMT2"] * 0.10


def test_consistency_can_be_switched_off(scorer: CreditScorer):
    raw = CounterfactualExplainer(scorer, consistency=NoConsistency())
    result = raw.explain(STRESSED, RecourseTarget(band="Low"))
    guarded = ArrearsConsistency().repair(STRESSED, STRESSED | {"PAY_1": 0})
    assert result.is_feasible
    assert guarded["PAY_AMT1"] == STRESSED["BILL_AMT2"]


def test_nothing_is_actionable_under_an_empty_policy(explainer: CounterfactualExplainer):
    locked = explainer.explain(STRESSED, RecourseTarget(band="Low"), RecoursePolicy({}))
    assert not locked.is_feasible
    assert locked.feature_changes == {}


# --- strategies and fallback ----------------------------------------------


def test_the_rule_based_search_finds_recourse_on_its_own(scorer: CreditScorer):
    fast = CounterfactualExplainer(scorer, search=RuleBasedSearch())
    result = fast.explain(STRESSED, RecourseTarget(band="Low"))
    assert result.is_feasible
    assert result.method == "rule-based"


def test_a_failing_primary_search_falls_back(scorer: CreditScorer):
    class Broken(GreedySearch):
        name = "broken"

        def search(self, *args, **kwargs):
            raise RuntimeError("optimizer unavailable")

    result = CounterfactualExplainer(scorer, search=Broken()).explain(STRESSED)
    assert result.is_feasible
    assert result.method == "rule-based"
    assert any("optimizer unavailable" in note for note in result.notes)


def test_a_timeout_falls_back_rather_than_returning_nothing(scorer: CreditScorer):
    result = CounterfactualExplainer(scorer, timeout_seconds=0.0).explain(STRESSED)
    assert result.method == "rule-based"
    assert result.is_feasible


def test_a_customer_already_in_the_target_band_needs_no_change(explainer):
    result = explainer.explain(example_customer(), RecourseTarget(band="Low"))
    assert result.is_feasible
    assert result.feature_changes == {}
    assert any("no change required" in note for note in result.notes)


# --- contract -------------------------------------------------------------


def test_an_incomplete_customer_is_rejected(explainer: CounterfactualExplainer):
    with pytest.raises(ValueError, match="Missing required columns"):
        explainer.explain({"LIMIT_BAL": 100_000})


def test_an_unknown_target_band_is_rejected(explainer: CounterfactualExplainer):
    with pytest.raises(ValueError, match="No ceiling for band"):
        explainer.explain(STRESSED, RecourseTarget(band="Risky"))


def test_an_explicit_probability_target_is_honoured(explainer: CounterfactualExplainer):
    result = explainer.explain(STRESSED, RecourseTarget(max_probability=0.20))
    assert result.target_score == 0.20
    if result.is_feasible:
        assert result.achieved_score <= 0.20


def test_the_result_serialises_for_the_api(recourse):
    payload = recourse.to_dict()
    assert set(payload) >= {
        "original_score", "target_score", "feature_changes", "is_feasible", "method"
    }
    assert all(set(change) == {"from", "to", "delta"} for change in payload["feature_changes"].values())
