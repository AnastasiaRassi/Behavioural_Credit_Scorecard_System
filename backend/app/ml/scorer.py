"""Scoring core: turn a customer's six months of history into a risk decision.

This is the single entry point the API and dashboard use. It loads the artifacts
written by `app.ml.train`, applies the same feature engineering used in training,
and returns a calibrated probability plus a risk band.
"""
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.feature_engineering import (
    RAW_COLUMNS,
    PROTECTED,
    engineer,
    clean,
    project_root,
)


@dataclass
class ScoreResult:
    probability: float
    risk_band: str
    threshold: float
    flagged: bool
    band_default_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


class ArtifactsMissing(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load():
    path = project_root() / "backend" / "artifacts"
    model_path = path / "model_ensemble.joblib"
    calibrator_path = path / "calibrator_isotonic.joblib"
    config_path = path / "scoring_config.json"

    missing = [p.name for p in (model_path, calibrator_path, config_path) if not p.exists()]
    if missing:
        raise ArtifactsMissing(
            f"Missing artifacts {missing} in {path}. Run: python -m app.ml.train"
        )

    models = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    return models, calibrator, config


class CreditScorer:
    """Scores customers. Cheap to construct; artifacts are loaded once and cached."""

    def __init__(self):
        self.models, self.calibrator, self.config = _load()
        self.feature_names = self.config["feature_names"]
        self.threshold = self.config["threshold_f1_optimal"]
        self.band_edges = self.config["band_edges"]
        self.band_report = self.config.get("band_report", {})

    # ---- input handling --------------------------------------------------
    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Engineer features and drop protected attributes.

        Public because the Explainer needs the exact frame the model sees. Every
        path that reaches a model must go through here.
        """
        missing = [c for c in RAW_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        features = engineer(clean(frame[RAW_COLUMNS].copy()))
        # Protected attributes are excluded from the decision by design.
        features = features.drop(columns=PROTECTED, errors="ignore")

        unexpected = set(features.columns) ^ set(self.feature_names)
        if unexpected:
            raise ValueError(
                f"Feature mismatch against the trained model: {sorted(unexpected)}. "
                "Retrain with python -m app.ml.train"
            )
        return features[self.feature_names]

    # ---- scoring ---------------------------------------------------------
    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.prepare(frame)
        raw = np.mean([m.predict_proba(features)[:, 1] for m in self.models], axis=0)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def band(self, probability: float) -> str:
        if probability < self.band_edges["low_max"]:
            return "Low"
        if probability < self.band_edges["medium_max"]:
            return "Medium"
        return "High"

    def score_one(self, customer: dict) -> ScoreResult:
        frame = pd.DataFrame([customer])
        probability = float(self.probabilities(frame)[0])
        band = self.band(probability)
        return ScoreResult(
            probability=round(probability, 4),
            risk_band=band,
            threshold=round(self.threshold, 4),
            flagged=probability >= self.threshold,
            band_default_rate=self.band_report.get(band, {}).get("actual_default_rate", float("nan")),
        )

    def score_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.probabilities(frame)
        return pd.DataFrame(
            {
                "probability": probabilities.round(4),
                "risk_band": [self.band(p) for p in probabilities],
                "flagged": probabilities >= self.threshold,
            },
            index=frame.index,
        )


def example_customer() -> dict:
    """A syntactically valid request, useful for smoke tests and API docs."""
    return {
        "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_1": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 40000, "BILL_AMT2": 38000, "BILL_AMT3": 36000,
        "BILL_AMT4": 34000, "BILL_AMT5": 32000, "BILL_AMT6": 30000,
        "PAY_AMT1": 3000, "PAY_AMT2": 3000, "PAY_AMT3": 3000,
        "PAY_AMT4": 3000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
    }
