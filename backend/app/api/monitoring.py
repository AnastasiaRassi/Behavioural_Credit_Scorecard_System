"""Monitoring: has the incoming population moved, and is the model still right.

Both endpoints report what the stored data can actually support. Neither invents
a number when there are too few rows, because a fabricated monitoring panel is
worse than an empty one.
"""
import pandas as pd
from fastapi import APIRouter, Depends
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.db.models import ObservedOutcome, Prediction
from app.ml import drift

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Below this many labelled predictions, production metrics are noise.
MIN_LABELLED = 100


@router.get("/drift")
def population_drift(session: Session = Depends(get_session)) -> dict:
    """PSI per feature, training split against everything scored since."""
    rows = [prediction.raw_input for prediction in session.query(Prediction).all()]
    live = pd.DataFrame(rows) if rows else pd.DataFrame()
    return drift.compare(live)


@router.get("/performance")
def production_performance(session: Session = Depends(get_session)) -> dict:
    """Metrics over predictions whose real outcome is now known.

    This is the only place production performance can come from: it needs labels,
    which arrive a month after the score. Until enough have, the endpoint says so
    rather than reporting a figure from a handful of rows.
    """
    pairs = (
        session.query(Prediction, ObservedOutcome)
        .join(ObservedOutcome, ObservedOutcome.prediction_id == Prediction.id)
        .all()
    )

    if len(pairs) < MIN_LABELLED:
        return {
            "sufficient": False,
            "labelled_predictions": len(pairs),
            "minimum_rows": MIN_LABELLED,
            "reason": (
                f"{len(pairs)} predictions have a known outcome; at least "
                f"{MIN_LABELLED} are needed before a metric means anything."
            ),
        }

    probabilities = [prediction.probability for prediction, _ in pairs]
    actuals = [int(outcome.defaulted) for _, outcome in pairs]
    flags = [int(prediction.flagged) for prediction, _ in pairs]

    return {
        "sufficient": True,
        "labelled_predictions": len(pairs),
        "default_rate": round(sum(actuals) / len(actuals), 4),
        "roc_auc": round(roc_auc_score(actuals, probabilities), 4),
        "f1_default_class": round(f1_score(actuals, flags), 4),
        "brier": round(brier_score_loss(actuals, probabilities), 4),
    }
