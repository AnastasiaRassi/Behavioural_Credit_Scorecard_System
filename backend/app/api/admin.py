"""The ML engineer's read-only surface: metrics, thresholds, importance, versions.

Every number here is read from artifacts/, so the API cannot report a metric that
differs from what the dashboard shows or from what was actually evaluated.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_explainer, get_scorer, get_session
from app.db.models import ModelVersion
from app.ml import artifacts
from app.schemas.scoring import CustomerInput, ReasonOut

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
        "split_sizes": config["split_sizes"],
        "deciles": config["deciles"],
        "calibration": config["calibration"],
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


@router.get("/segments")
def behavioural_segments() -> dict:
    """The behavioural segmentation: chosen k, how it was chosen, each profile.

    Separate from the model. Segments describe how customers behave; the model
    predicts whether they will miss a payment. The segment is never a model
    input, so this is a lens on the population rather than part of the decision.

    503 when segmentation has not been run, since the model works without it.
    """
    try:
        config = artifacts.segments()
    except artifacts.ArtifactsMissing as exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exception)
        ) from exception

    return {
        "k": config["k"],
        "chosen_by": config["chosen_by"],
        "silhouette_scores": config["silhouette_scores"],
        "features": config["features"],
        "fitted_on": config["fitted_on"],
        "segments": config["segments"],
    }


@router.post("/explain")
def explain_without_storing(customer: CustomerInput) -> dict:
    """Score and explain a hypothetical customer without recording anything.

    The engineer's what-if panel. Separate from POST /predictions because an
    experiment is not a decision: writing every slider movement into the audit
    trail would bury the real ones. Returns every reason above the negligible
    threshold rather than the handful a screen shows.
    """
    payload = customer.model_dump()
    result = get_scorer().score_one(payload)
    reasons, base_value = get_explainer().reasons(payload)

    return {
        "stored": False,
        "probability": result.probability,
        "risk_band": result.risk_band,
        "threshold": result.threshold,
        "flagged": result.flagged,
        "band_default_rate": result.band_default_rate,
        "base_value": base_value,
        "reasons": [
            ReasonOut(**reason.to_dict(), raises_risk=reason.raises_risk).model_dump()
            for reason in reasons
        ],
    }


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
