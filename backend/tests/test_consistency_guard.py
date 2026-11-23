"""The consistency guard: pure rules, the scorer, and the customers the model gets wrong."""
import json
from pathlib import Path

import pytest

from app.ml import consistency_guard as guard
from app.ml.exceptions import ArtifactsMissing
from app.ml.scorer import CreditScorer, example_customer

CASES = json.loads((Path(__file__).parent / "fixtures" / "failure_cases.json").read_text())["cases"]


def statuses(*values):
    return list(values) + [0] * (6 - len(values))


@pytest.mark.parametrize(
    "band, pay, expected_band, flag",
    [
        ("Low", [1, 1, 1, 0, 0, 0], "Medium", guard.PERSISTENT_DELAY_FLOOR),
        ("Low", [2, 0, 0, 0, 0, 0], "Medium", guard.CURRENT_DELAY_FLOOR),
        ("Low", [1] * 6, "Medium", guard.PERSISTENT_DELAY_FLOOR),
        ("Medium", [0, 0, 2, 2, 2, 2], "Medium", guard.DEEP_DELAY_REVIEW),
        ("Medium", [2, 0, 0, 0, 0, 0], "Medium", guard.DEEP_DELAY_REVIEW),
    ],
)
def test_rules_fire(band, pay, expected_band, flag):
    result = guard.check(band, pay)
    assert result.band == expected_band
    assert flag in result.flags
    assert result.needs_review


@pytest.mark.parametrize(
    "band, pay",
    [
        ("Low", [0] * 6),
        ("Low", [1, 1, 0, 0, 0, 0]),
        ("Low", [-2] * 6),
        ("Medium", [1, 1, 1, 0, 0, 0]),
        ("High", [2] * 6),
        ("High", [-1] * 6),
    ],
)
def test_clean_or_defensible_bands_are_left_alone(band, pay):
    result = guard.check(band, pay)
    assert result == guard.GuardResult(band, band)


def test_guard_never_lowers_a_band():
    for band in guard.BANDS:
        for pay in ([2] * 6, [0] * 6, [-2] * 6, [1, 0, 1, 0, 1, 0]):
            result = guard.check(band, pay)
            assert guard.BANDS.index(result.band) >= guard.BANDS.index(band)


def test_bad_input_is_reviewed_not_reshaped():
    record = example_customer()
    record["PAY_1"] = 15
    assert guard.input_problems(record)
    result = guard.check_record("Low", record)
    assert result.band == "Low" and result.flags == (guard.INVALID_INPUT,) and result.needs_review


@pytest.mark.parametrize("field, value", [("PAY_AMT1", -5), ("LIMIT_BAL", 0), ("BILL_AMT3", float("nan")), ("PAY_2", None)])
def test_input_problems_catch_malformed_fields(field, value):
    record = example_customer()
    record[field] = value
    assert guard.input_problems(record)


def test_valid_example_customer_has_no_problems():
    assert guard.input_problems(example_customer()) == []


def test_apply_to_row_is_idempotent():
    row = {"risk_band": "Low", "history": [{"month": m, "payment_status": 2} for m in range(1, 7)]}
    once = guard.apply_to_row(row)
    assert once["risk_band"] == "Medium" and once["model_band"] == "Low"
    assert guard.apply_to_row(once) == once


def scorer_or_skip():
    try:
        return CreditScorer()
    except ArtifactsMissing as exception:
        pytest.skip(str(exception))


def test_scorer_raises_a_persistently_late_low_score():
    scorer = scorer_or_skip()
    customer = example_customer()
    customer.update({f"PAY_{i}": 1 for i in range(1, 7)})
    result = scorer.score_one(customer)
    if result.model_band != "Low":
        pytest.skip("model no longer scores this profile Low")
    assert result.risk_band == "Medium" and result.needs_review
    assert result.guard_flags == (guard.PERSISTENT_DELAY_FLOOR,)


def test_scorer_leaves_a_normal_customer_alone():
    result = scorer_or_skip().score_one(example_customer())
    assert result.risk_band == result.model_band and not result.needs_review


def test_batch_matches_single_scoring():
    import pandas as pd

    scorer = scorer_or_skip()
    late = example_customer()
    late.update({f"PAY_{i}": 1 for i in range(1, 7)})
    frame = pd.DataFrame([example_customer(), late])
    batch = scorer.score_batch(frame)
    for position, customer in enumerate((example_customer(), late)):
        single = scorer.score_one(customer)
        assert batch.iloc[position]["risk_band"] == single.risk_band
        assert bool(batch.iloc[position]["needs_review"]) == single.needs_review


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["pattern"])
def test_known_failure_customers(case):
    from app.ml import worklist

    try:
        rows = {row["account_number"]: row for row in worklist.all_rows()}
    except ArtifactsMissing as exception:
        pytest.skip(str(exception))
    for account in case["accounts"]:
        row = rows[str(account)]
        assert row["risk_band"] == case["band"], account
        assert row["needs_review"] == case["review"], account
