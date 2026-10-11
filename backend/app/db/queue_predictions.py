"""Bridge between the precomputed worklist and the decision trail.

Queue rows live in an artifact, decisions live in the database and point at a
prediction row. So the first time an officer acts on a queued customer, that
customer's score has to become a real prediction. This module does that and
nothing else.

The score is taken from the queue rather than recomputed. It is the number the
officer was looking at when they decided, and re-scoring could not reproduce it
anyway: the queue is built from the blind feature set, so the protected columns
`CreditScorer.prepare()` expects are not present.
"""
from sqlalchemy.orm import Session

from app.db.models import ModelVersion, Prediction, User
from app.ml.explainability import MONTHS_OBSERVED


def raw_input_from_history(row: dict) -> dict:
    """The submitted columns behind a queue row, as far as they are known.

    Nineteen of the dataset's twenty-three columns. The four protected
    attributes are absent because the queue is built blind, and an officer never
    saw them, so recording them here would overstate what the decision was based
    on.
    """
    raw = {"LIMIT_BAL": row["credit_limit"]}
    for entry in row.get("history") or []:
        position = entry["month"]
        raw[f"PAY_{position}"] = entry["payment_status"]
        raw[f"BILL_AMT{position}"] = entry["bill_amount"]
        raw[f"PAY_AMT{position}"] = entry["amount_paid"]

    raw["_source"] = "officer_queue"
    raw["_protected_attributes_withheld"] = True
    raw["_months_observed"] = MONTHS_OBSERVED
    return raw


def cardholder_for(session: Session, customer_id: str) -> User:
    """The cardholder row for a queue customer, created on first contact."""
    existing = (
        session.query(User)
        .filter_by(display_name=customer_id, role="cardholder")
        .one_or_none()
    )
    if existing is not None:
        return existing

    cardholder = User(display_name=customer_id, role="cardholder")
    session.add(cardholder)
    session.flush()
    return cardholder


def scored_columns(row: dict) -> dict:
    """Just the model's inputs, without the bookkeeping keys."""
    return {
        key: value
        for key, value in raw_input_from_history(row).items()
        if not key.startswith("_")
    }


def latest_for_cardholder(session: Session, customer_id: str) -> Prediction | None:
    """The newest stored score for a queue customer, if one has been run."""
    cardholder = (
        session.query(User)
        .filter_by(display_name=customer_id, role="cardholder")
        .one_or_none()
    )
    if cardholder is None:
        return None

    return (
        session.query(Prediction)
        .filter_by(subject_user_id=cardholder.id)
        .order_by(Prediction.scored_at.desc())
        .first()
    )


def store_fresh_prediction(
    session: Session,
    row: dict,
    version: ModelVersion,
    probability: float,
    risk_band: str,
    threshold: float,
    reasons: list[dict],
    base_value: float,
) -> Prediction:
    """Record a score the officer asked for, as a new row.

    Appends rather than replacing, so re-running the prediction leaves the
    earlier one in place: a decision taken last week still points at the score
    that was on screen then.
    """
    prediction = Prediction(
        subject_user_id=cardholder_for(session, row["customer_id"]).id,
        model_version_id=version.id,
        probability=probability,
        risk_band=risk_band,
        flagged=probability >= threshold,
        threshold=threshold,
        raw_input=raw_input_from_history(row),
        explanation={
            "base_value": base_value,
            "reasons": reasons,
            "source": "officer_request",
        },
    )
    session.add(prediction)
    session.flush()
    return prediction


def prediction_for(
    session: Session, row: dict, version: ModelVersion, threshold: float
) -> Prediction:
    """The prediction row for a queued customer, materialised once.

    Reused on later decisions rather than duplicated, so a customer monitored
    last week and escalated today has one score with two decisions against it,
    which is what the audit trail is supposed to show.
    """
    cardholder = cardholder_for(session, row["customer_id"])

    existing = (
        session.query(Prediction)
        .filter_by(subject_user_id=cardholder.id, model_version_id=version.id)
        .order_by(Prediction.scored_at.desc())
        .first()
    )
    if existing is not None:
        return existing

    prediction = Prediction(
        subject_user_id=cardholder.id,
        model_version_id=version.id,
        probability=row["probability"],
        risk_band=row["risk_band"],
        flagged=row["flagged"],
        threshold=threshold,
        raw_input=raw_input_from_history(row),
        explanation={"reasons": row["reasons"], "source": "officer_queue"},
    )
    session.add(prediction)
    session.flush()
    return prediction
