"""The ML engineer's read-only surface: metrics, thresholds, importance, versions.

Every number here is read from artifacts/, so the API cannot report a metric that
differs from what the dashboard shows or from what was actually evaluated.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.db.models import ModelVersion
from app.ml import artifacts

router = APIRouter(tags=["model"])


@router.get("/report")
def model_report() -> dict:
    """Headline metrics and operating points for the served model."""
    config = artifacts.scoring_config()
    return {
        "model_version": config["model_version"],
        "trained": config["trained"],
        "scale": config["scale"],
        "training_rows": config["training_rows"],
        "test_rows": config["test_rows"],
        "test_default_rate": config["test_default_rate"],
        "n_features": config["n_features"],
        "seeds": config["seeds"],
        "threshold": config["threshold_f1_optimal"],
        "band_edges": config["band_edges"],
        "test_metrics": config["test_metrics"],
        "confusion_matrix": config["confusion_matrix"],
        "band_report": config["band_report"],
    }


@router.get("/report/threshold-sweep")
def threshold_sweep() -> list[dict]:
    """Precision, recall and counts at every swept threshold."""
    return artifacts.threshold_sweep()


@router.get("/report/cost-curve")
def cost_curve() -> dict:
    """Expected cost across threshold choices, by false-negative cost ratio."""
    return artifacts.cost_curve()


@router.get("/report/feature-importance")
def feature_importance() -> list[dict]:
    """Permutation importance over the test split, most important first."""
    return artifacts.feature_importance()


@router.get("/model-versions")
def model_versions(session: Session = Depends(get_session)) -> list[dict]:
    """Registered versions, with the artifact registry alongside.

    The database row carries identity and which model is live; the metrics come
    from artifacts/, which stays the single source of truth for them.
    """
    registered = session.query(ModelVersion).order_by(
        ModelVersion.trained_on.desc()
    ).all()
    recorded = {entry["version"]: entry for entry in artifacts.registry()}

    return [
        {
            "version": version.version,
            "trained_on": version.trained_on.isoformat(),
            "is_live": version.is_live,
            "registered_at": version.registered_at.isoformat(),
            "metrics": recorded.get(version.version),
        }
        for version in registered
    ]
