"""Concept reasons: every feature grouped once, sums exact, wording free of raw codes."""
import numpy as np
import pandas as pd
import pytest

from app.ml.explainability import CONCEPTS, concept_of, concept_reasons
from app.ml.feature_engineering import RAW_COLUMNS, clean, engineer

PAY = [f"PAY_{i}" for i in range(1, 7)]


def _row(**raw) -> pd.Series:
    base = {"LIMIT_BAL": 50_000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 34,
            **{p: 0 for p in PAY},
            **{f"BILL_AMT{i}": 10_000 for i in range(1, 7)},
            **{f"PAY_AMT{i}": 10_000 for i in range(1, 7)}}
    base.update(raw)
    frame = pd.DataFrame([{c: base[c] for c in RAW_COLUMNS}])
    return engineer(clean(frame)).drop(columns=["SEX", "EDUCATION", "MARRIAGE", "AGE"]).iloc[0]


def _reasons(row: pd.Series, effects: dict[str, float] | None = None):
    effects = effects or {}
    vector = np.array([effects.get(c, 0.05) for c in row.index])
    return {r.feature: r for r in concept_reasons(row, vector, row.index)}


def test_every_served_feature_has_exactly_one_concept():
    row = _row()
    assert {concept_of(c) for c in row.index} == set(CONCEPTS)


def test_concept_effect_is_the_exact_sum_of_its_parts():
    row = _row()
    reasons = _reasons(row)
    for reason in reasons.values():
        assert reason.effect == pytest.approx(sum(p["effect"] for p in reason.parts))
    total = sum(r.effect for r in reasons.values())
    assert total == pytest.approx(0.05 * len(row.index))


def test_status_codes_are_described_as_what_happened():
    # Charbel's pattern: never behind, a mix of paid in full and carried over.
    row = _row(PAY_1=0, PAY_2=0, PAY_3=-1, PAY_4=-1, PAY_5=-1, PAY_6=0)
    text = _reasons(row)["repayment_status"].description
    assert "revolving" not in text.lower()
    assert "never behind" in text and "paid in full in 3 months" in text


def test_arrears_name_the_count_and_the_worst_delay():
    row = _row(PAY_1=2, PAY_2=1, PAY_3=0)
    text = _reasons(row)["repayment_status"].description
    assert "behind in 2 of the last 6 months" in text and "2 months late" in text


def test_payment_amounts_are_not_read_as_underpaying():
    # Pavel's pattern: tiny, mostly idle account, every bill cleared.
    row = _row(BILL_AMT1=4166, BILL_AMT2=0, BILL_AMT3=4166, BILL_AMT4=0, BILL_AMT5=0,
               BILL_AMT6=0, PAY_AMT1=0, PAY_AMT2=4166, PAY_AMT3=0, PAY_AMT4=0,
               PAY_AMT5=0, PAY_AMT6=0)
    reasons = _reasons(row, {"bill_mean": 0.2, "pay_amt_mean": 0.2})
    size = reasons["account_size"].description
    assert "mostly inactive" in size and "paid" not in size and "average payment" not in size
    assert "less active accounts miss payments more often" in size
    assert "cleared the one bill due" in reasons["share_paid"].description


def test_over_the_limit_is_said_plainly():
    row = _row(BILL_AMT1=60_000)
    assert "over the limit by" in _reasons(row)["credit_in_use"].description


def test_weak_concepts_are_left_out():
    row = _row()
    reasons = concept_reasons(row, np.zeros(len(row.index)), row.index)
    assert reasons == []
