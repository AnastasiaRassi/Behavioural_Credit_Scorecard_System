"""Checks for the plain-language wording in describe(): no raw codes, no stray decimals."""
import re

import pytest

from app.ml import artifacts
from app.ml.explainability import describe


def test_pay_mean_is_a_banded_phrase_with_no_decimal():
    assert describe("pay_mean", 4.7) == "Usually behind on repayments across the six months"
    assert describe("pay_mean", 0.5) == "Sometimes behind on repayments across the six months"
    assert describe("pay_mean", -1.3) == "Usually up to date on repayments across the six months"


def test_pay_mean_cut_offs_sit_on_the_pay_status_scale():
    # 1 is a month in arrears, 0 is revolving: the bands split on those two values.
    assert describe("pay_mean", 1.0).startswith("Usually behind")
    assert describe("pay_mean", 0.99).startswith("Sometimes behind")
    assert describe("pay_mean", 0.0).startswith("Usually up to date")
    assert describe("pay_mean", 0.01).startswith("Sometimes behind")


def test_a_delay_longer_than_the_window_is_kept_as_months():
    # PAY_x is months of delay and may exceed the six observed, so 7 is valid.
    assert describe("PAY_1", 7) == "Last month: 7 months in arrears"
    assert describe("pay_max", 8) == "Worst month in the six: 8 months in arrears"


def test_pay_max_with_no_arrears_says_never_behind():
    assert describe("pay_max", -2) == "Never behind on payments in the last six months"
    assert describe("pay_max", 0) == "Never behind on payments in the last six months"


def test_util_trend_says_which_way_it_moved_instead_of_a_bare_share():
    assert describe("util_trend", 0.12) == "Share of the limit in use is up 12 percentage points since six months ago"
    assert describe("util_trend", -0.3).startswith("Share of the limit in use is down 30")
    assert "unchanged" in describe("util_trend", 0.0)
    assert "not available" in describe("util_trend", float("nan"))


def test_pay_to_bill_gap_keeps_its_sign_in_words():
    assert describe("pay_to_bill_gap", 1500) == "Payments fall short of the average bill by 1,500 a month"
    assert describe("pay_to_bill_gap", -200) == "Payments exceed the average bill by 200 a month"
    assert describe("pay_to_bill_gap", 0) == "Payments match the average bill"


@pytest.mark.parametrize("value", [0.0, 0.4567, 2.5, 7.0, -1.5])
def test_no_model_feature_is_described_with_a_decimal_number(value):
    for feature in artifacts.scoring_config()["feature_names"]:
        text = describe(feature, value)
        assert not re.search(r"\d\.\d", text), f"{feature}: {text}"
        # The generic fallback prints "name: value" and no feature should reach it.
        assert not re.search(r"\b[a-z]+_[a-z_0-9]+\b", text), f"{feature}: {text}"
