"""Smoke checks for cardholder coaching: the template always returns text built only from its payload.

Run from the backend/ directory:

    python -m pytest tests/test_coaching.py
"""
from app.ai import coaching
from app.ml import artifacts
from app.ml.scorer import example_customer

# Two months in arrears and nothing paid against a near-full card: the model rates
# this account High, and one month's payment can move it to Medium.
STRESSED = example_customer() | {
    "PAY_1": 2, "PAY_2": 2, "PAY_3": 1,
    "BILL_AMT1": 195_000, "PAY_AMT1": 0, "PAY_AMT2": 0,
}

REASONS = [
    {"feature": "PAY_1", "description": "Last month: 2 months in arrears", "effect": 0.9},
    {"feature": "util_mean", "description": "Average share of the limit used at 95%", "effect": 0.3},
    {"feature": "PAY_AMT1", "description": "No payment made in month 1", "effect": 0.2},
    {"feature": "LIMIT_BAL", "description": "Credit limit of 200,000", "effect": -0.1},
]


def test_returns_text_and_every_number_is_in_the_payload():
    result = coaching.coach("TEST-HIGH", "High", REASONS, STRESSED)

    assert result.text.strip()
    assert result.disclaimer == coaching.DISCLAIMER
    assert coaching.is_grounded(result.text, coaching.build_payload("High", REASONS, STRESSED))


def test_lowest_band_needs_no_search_and_says_so():
    assert "lowest risk band" in coaching.coach("TEST-LOW", "Low", REASONS, None).text
    assert coaching.build_payload("Low", REASONS).outcome == coaching.LOWEST_BAND


def test_missing_record_is_reported_not_guessed():
    payload = coaching.build_payload("High", REASONS, None)

    assert payload.outcome == coaching.UNCHECKED
    assert "could not be checked" in coaching.render(payload)


def test_unreachable_target_says_no_single_month_action_clears_it():
    payload = coaching.CoachingPayload(
        band="High", reasons=("Last month: 2 months in arrears",), target_band="Medium",
        outcome=coaching.UNREACHABLE, steps=(),
    )

    text = coaching.render(payload)

    assert "No single-month action clears it" in text
    assert "Medium risk" in text


def test_reachable_target_lists_the_steps_from_the_payload():
    payload = coaching.build_payload("High", REASONS, STRESSED)

    assert payload.outcome == coaching.REACHABLE
    assert payload.steps
    text = coaching.render(payload)
    for step in payload.steps:
        assert step in text


def test_payload_is_memoised_for_the_same_account():
    first = coaching.build_payload("High", REASONS, STRESSED)
    assert coaching.build_payload("High", REASONS, STRESSED) is first


def test_cache_is_used_only_while_model_version_and_account_match(monkeypatch):
    entry = coaching.cache_entry("TEST-HIGH", "High", REASONS, STRESSED)
    entry["coaching"]["text"] = "cached text"
    current = {"model_version": coaching._model_version(), "entries": {"TEST-HIGH": entry}}
    monkeypatch.setattr(artifacts, "coaching_cache", lambda: current)

    assert coaching.coach("TEST-HIGH", "High", REASONS, STRESSED).text == "cached text"

    stale = {**current, "model_version": "an_older_model"}
    monkeypatch.setattr(artifacts, "coaching_cache", lambda: stale)
    assert coaching.coach("TEST-HIGH", "High", REASONS, STRESSED).text != "cached text"
