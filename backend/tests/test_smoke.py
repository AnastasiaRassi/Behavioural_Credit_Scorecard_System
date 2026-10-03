"""Smoke checks: assert the served model against known values.

Formalises the practice described in docs/CODE_CONVENTIONS.md §14 — every module
gained a smoke check asserting a known number before being considered done. These
are those checks, as the first cases of a real suite.

Run from the backend/ directory:

    python -m pytest tests/

They read the artifacts rather than retraining, so they are fast, and they fail
loudly if a retrain moves a published figure.
"""
import json

import pandas as pd
import pytest

from app.ml.explainability import Explainer, describe, feature_meaning
from app.ml.scorer import CreditScorer, example_customer
from app.ml.train import artifacts_dir

# Known values for the current model. Update deliberately, never to make a test pass.
EXPECTED_AUC = 0.7811
EXPECTED_F1 = 0.5403
EXPECTED_BRIER = 0.1349
EXPECTED_THRESHOLD = 0.2747
TOLERANCE = 0.0005

EXPECTED_FEATURES = 46
EXPECTED_TEST_ROWS = 4500
EXPECTED_TRAINING_ROWS = 25500

ARTIFACT_FILES = [
    "model_ensemble.joblib",
    "calibrator_isotonic.joblib",
    "scoring_config.json",
    "threshold_sweep.json",
    "feature_importance.json",
    "cost_curve.json",
    "fairness_detail.json",
    "registry.json",
]


@pytest.fixture(scope="module")
def config() -> dict:
    with open(artifacts_dir() / "scoring_config.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def scorer() -> CreditScorer:
    return CreditScorer()


# --- the artifact contract ------------------------------------------------


@pytest.mark.parametrize("name", ARTIFACT_FILES)
def test_artifact_exists(name: str):
    assert (artifacts_dir() / name).exists(), f"{name} missing; run app.ml.train"


def test_headline_metrics_match_known_values(config: dict):
    metrics = config["test_metrics"]
    assert abs(metrics["roc_auc"] - EXPECTED_AUC) < TOLERANCE
    assert abs(metrics["f1"] - EXPECTED_F1) < TOLERANCE
    assert abs(metrics["brier"] - EXPECTED_BRIER) < TOLERANCE
    assert abs(config["threshold_f1_optimal"] - EXPECTED_THRESHOLD) < TOLERANCE


def test_gini_is_consistent_with_auc(config: dict):
    metrics = config["test_metrics"]
    assert abs(metrics["gini"] - (2 * metrics["roc_auc"] - 1)) < 1e-9


def test_split_shapes(config: dict):
    assert config["n_features"] == EXPECTED_FEATURES
    assert config["test_rows"] == EXPECTED_TEST_ROWS
    assert config["training_rows"] == EXPECTED_TRAINING_ROWS


def test_confusion_matrix_sums_to_the_test_split(config: dict):
    counts = config["confusion_matrix"]
    assert sum(counts.values()) == config["test_rows"]


def test_dashboard_panels_are_present(config: dict):
    assert len(config["deciles"]) == 10
    assert config["calibration"], "calibration curve is empty"
    # Deciles must be ordered safest first for the chart to read correctly.
    rates = [decile["actual_default_rate"] for decile in config["deciles"]]
    assert rates[0] < rates[-1]


# --- fairness -------------------------------------------------------------


def test_every_protected_attribute_passes_four_fifths():
    with open(artifacts_dir() / "fairness_detail.json", encoding="utf-8") as fh:
        audit = json.load(fh)

    assert set(audit) == {"SEX", "AGE", "EDUCATION", "MARRIAGE"}
    for attribute, values in audit.items():
        assert values["passes_four_fifths"], f"{attribute} fails the four-fifths rule"
        assert values["disparate_impact_ratio"] >= 0.80


def test_small_groups_are_hidden_rather_than_quoted():
    with open(artifacts_dir() / "fairness_detail.json", encoding="utf-8") as fh:
        audit = json.load(fh)

    for values in audit.values():
        for group in values["groups"]:
            if not group["reliable"]:
                assert group["group"] in values["hidden_groups"]


# --- scoring --------------------------------------------------------------


def test_protected_attributes_never_reach_the_model(scorer: CreditScorer):
    features = scorer.prepare(pd.DataFrame([example_customer()]))
    for attribute in ("SEX", "AGE", "EDUCATION", "MARRIAGE"):
        assert attribute not in features.columns


def test_a_stressed_customer_scores_higher_than_a_healthy_one(scorer: CreditScorer):
    healthy = example_customer()
    stressed = example_customer() | {
        "PAY_1": 3, "PAY_2": 3, "PAY_3": 2,
        "BILL_AMT1": 195_000, "PAY_AMT1": 0, "PAY_AMT2": 0,
    }
    assert scorer.score_one(stressed).probability > scorer.score_one(healthy).probability


def test_bands_follow_the_saved_edges(scorer: CreditScorer):
    edges = scorer.band_edges
    assert scorer.band(edges["low_max"] - 0.01) == "Low"
    assert scorer.band(edges["medium_max"] - 0.01) == "Medium"
    assert scorer.band(edges["medium_max"] + 0.01) == "High"


# --- explanations ---------------------------------------------------------


def test_reasons_sum_to_the_raw_score(scorer: CreditScorer):
    """The explanation must be complete, not a summary of the top few drivers."""
    explainer = Explainer(scorer)
    customer = example_customer() | {"PAY_1": 2, "PAY_AMT1": 0}

    features = scorer.prepare(pd.DataFrame([customer]))
    effects, base = explainer._contributions(features)
    raw = scorer.models[0].predict_proba(features, raw_score=True)[0]

    # Compared against one model, since effects are averaged over the ensemble.
    assert abs(float(effects.sum()) + base - float(raw)) < 0.05


def test_reasons_are_ordered_by_strength(scorer: CreditScorer):
    explainer = Explainer(scorer)
    reasons, _ = explainer.reasons(example_customer() | {"PAY_1": 2}, limit=5)

    assert reasons, "no reasons returned"
    effects = [abs(reason.effect) for reason in reasons]
    assert effects == sorted(effects, reverse=True)


def test_arrears_drive_risk_up(scorer: CreditScorer):
    explainer = Explainer(scorer)
    customer = example_customer() | {"PAY_1": 3, "PAY_2": 3, "PAY_AMT1": 0}
    reasons, _ = explainer.reasons(customer, limit=3)

    assert any(reason.raises_risk for reason in reasons)


def test_descriptions_avoid_raw_column_names():
    assert describe("PAY_AMT1", 0) == "No payment made in month 1"
    assert describe("pay_months_late", 0) == "Never in arrears in the last six months"
    assert "160,000" in describe("remaining_credit", 160_000)
    assert "PAY_" not in describe("PAY_1", 2)


def test_every_feature_has_a_plain_meaning(config: dict):
    for feature in config["feature_names"]:
        meaning = feature_meaning(feature)
        assert meaning and meaning != feature, f"{feature} has no plain meaning"
