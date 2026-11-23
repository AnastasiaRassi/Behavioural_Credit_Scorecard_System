"""Scoring core: turn a customer's six months of history into a risk decision.

This is the single entry point the API and dashboard use. It loads the artifacts
written by `app.ml.train`, applies the same feature engineering used in training,
and returns a probability plus a risk band.
"""
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.calibration import calibrate
from app.ml.consistency_guard import check_record
from app.ml.exceptions import (
    ArtifactCorrupt,
    ArtifactsMissing,
    FeatureSchemaMismatch,
    MissingColumns,
    ScoringFailed,
)
from app.ml.feature_engineering import (
    RAW_COLUMNS,
    SCORED_COLUMNS,
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
    # The guard may raise `risk_band`; the model's own band and the reasons are kept.
    model_band: str = ""
    guard_flags: tuple[str, ...] = ()
    needs_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _load():
    path = project_root() / "backend" / "artifacts"
    model_path = path / "model_ensemble.joblib"
    config_path = path / "scoring_config.json"

    missing = [p.name for p in (model_path, config_path) if not p.exists()]
    if missing:
        raise ArtifactsMissing(
            f"Missing artifacts {missing} in {path}. Run: python -m app.ml.train"
        )

    try:
        models = joblib.load(model_path)
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exception:
        raise ArtifactCorrupt(
            f"Cannot load model artifacts from {path}: {exception!r}. "
            "Re-run: python -m app.ml.train",
            context={"path": str(path)},
        ) from exception
    return models, config


class CreditScorer:
    """Scores customers. Cheap to construct; artifacts are loaded once and cached."""

    def __init__(self):
        self.models, self.config = _load()
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
        missing = [c for c in SCORED_COLUMNS if c not in frame.columns]
        if missing:
            raise MissingColumns(
                f"Missing required columns: {missing}", context={"columns": missing}
            )

        # Protected columns are taken when offered and ignored otherwise. They
        # are dropped below either way, so requiring them would block scoring a
        # blind frame for no benefit.
        present = [c for c in RAW_COLUMNS if c in frame.columns]
        features = engineer(clean(frame[present].copy()))
        # Protected attributes are excluded from the decision by design.
        features = features.drop(columns=PROTECTED, errors="ignore")

        unexpected = set(features.columns) ^ set(self.feature_names)
        if unexpected:
            raise FeatureSchemaMismatch(
                f"Feature mismatch against the trained model: {sorted(unexpected)}. "
                "Retrain with python -m app.ml.train",
                context={"features": sorted(unexpected)},
            )
        return features[self.feature_names]

    # ---- scoring ---------------------------------------------------------
    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.prepare(frame)
        try:
            raw = np.mean([m.predict_proba(features)[:, 1] for m in self.models], axis=0)
            return calibrate(raw)
        except Exception as exception:
            raise ScoringFailed(
                f"model inference failed on {len(features)} row(s): {exception!r}",
                context={"rows": len(features)},
            ) from exception

    def band(self, probability: float) -> str:
        if probability < self.band_edges["low_max"]:
            return "Low"
        if probability < self.band_edges["medium_max"]:
            return "Medium"
        return "High"

    def score_one(self, customer: dict) -> ScoreResult:
        frame = pd.DataFrame([customer])
        probability = float(self.probabilities(frame)[0])
        model_band = self.band(probability)
        verdict = check_record(model_band, customer)
        return ScoreResult(
            probability=round(probability, 4),
            risk_band=verdict.band,
            threshold=round(self.threshold, 4),
            flagged=probability >= self.threshold,
            # The rate belongs to the model's band, which is what the probability calibrates to.
            band_default_rate=self.band_report.get(model_band, {}).get("actual_default_rate", float("nan")),
            model_band=model_band,
            guard_flags=verdict.flags,
            needs_review=verdict.needs_review,
        )

    def score_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.probabilities(frame)
        model_bands = [self.band(p) for p in probabilities]
        verdicts = [
            check_record(band, record)
            for band, record in zip(model_bands, frame.to_dict("records"))
        ]
        return pd.DataFrame(
            {
                "probability": probabilities.round(4),
                "risk_band": [v.band for v in verdicts],
                "model_band": model_bands,
                "flagged": probabilities >= self.threshold,
                "needs_review": [v.needs_review for v in verdicts],
                "guard_flags": [",".join(v.flags) for v in verdicts],
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
