"""Bridge between the precomputed worklist and the decision trail.

Queue rows live in an artifact, decisions live in the database and point at a
prediction row. So the first time an officer acts on a queued customer, that
customer's score has to become a real prediction (`prediction_for`). An officer's
"predict default" instead stores a fresh live score (`store_fresh_prediction`);
the `latest_*` helpers read the newest stored score back.

A decision's score is taken from the queue rather than recomputed. It is the number the
officer was looking at when they decided, and re-scoring could not reproduce it
anyway: the queue is built from the blind feature set, so the protected columns
`CreditScorer.prepare()` expects are not present.

A prediction points straight at `customer.id`, so no cardholder row has to be
invented to carry an id. The customer must already be ingested; the queue is
built from the same dataset, so a missing one means ingestion has not been run.

ml/worklist.py already maps the artifact's row labels onto account numbers, so a
queue row's account number is the key to look up. Every lookup here still checks
the figures against the stored ones rather than trusting the match, because
pointing a decision at the wrong person's account is the worst thing this module
could do.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.customer_registry import find_by_account_number
from app.db.models import Customer, ModelVersion, MonthlyStatement, Prediction
from app.ml.explainability import MONTHS_OBSERVED


class CustomerNotIngested(RuntimeError):
    pass


class QueueIdMismatch(RuntimeError):
    pass


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


def scored_columns(row: dict) -> dict:
    """Just the model's inputs, without the bookkeeping keys."""
    return {
        key: value
        for key, value in raw_input_from_history(row).items()
        if not key.startswith("_")
    }


def customer_for_queue_row(session: Session, row: dict) -> Customer:
    """The stored customer a queue row describes, with the id mapping checked."""
    account_number = row["account_number"]
    customer = find_by_account_number(session, account_number)
    if customer is None:
        raise CustomerNotIngested(
            f"Queue row {account_number!r} has no customer in the database, so no "
            "prediction can reference it. Run: python -m app.db.ingest"
        )

    _verify_same_account(row, customer, session)
    return customer


def latest_for_cardholder(session: Session, customer_id: uuid.UUID) -> Prediction | None:
    """The newest stored score for a queue customer, if one has been run."""
    return session.scalars(
        select(Prediction)
        .where(Prediction.subject_customer_id == customer_id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).first()


def latest_by_customer(
    session: Session, customer_ids: list[uuid.UUID]
) -> dict[uuid.UUID, Prediction]:
    """The newest stored score per customer, for those that have one."""
    if not customer_ids:
        return {}
    latest: dict[uuid.UUID, Prediction] = {}
    for prediction in session.scalars(
        select(Prediction)
        .where(Prediction.subject_customer_id.in_(customer_ids))
        .order_by(Prediction.created_at.desc())
    ):
        latest.setdefault(prediction.subject_customer_id, prediction)
    return latest


def scored_account_numbers(session: Session) -> set[str]:
    """Account numbers with at least one stored prediction."""
    return set(
        session.scalars(
            select(Customer.account_number)
            .join(Prediction, Prediction.subject_customer_id == Customer.id)
            .distinct()
        )
    )


def _verify_same_account(row: dict, customer: Customer, session: Session) -> None:
    recent = session.get(MonthlyStatement, (customer.id, 1))
    stored = (
        float(customer.credit_limit),
        float(recent.bill_amount) if recent else None,
        float(recent.payment_amount) if recent else None,
    )
    queued = (
        float(row["credit_limit"]),
        float(row["current_balance"]),
        float(row["last_payment"]),
    )
    if stored != queued:
        raise QueueIdMismatch(
            f"Queue row {row['account_number']!r} maps to customer "
            f"{customer.id}, but their figures differ "
            f"(limit/bill/payment stored {stored} vs queued {queued}). The "
            "worklist artifact and the ingested data are out of step; "
            "regenerate it with: python -m app.ml.train"
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
    scored_by_user_id: uuid.UUID | None,
    source: str = "officer_request",
) -> Prediction:
    """Record a live score as a new row, naming the officer who asked (None if self-service).

    Appends rather than replacing, so re-running the prediction leaves the
    earlier one in place: a decision taken last week still points at the score
    that was on screen then.
    """
    customer = customer_for_queue_row(session, row)

    prediction = Prediction(
        subject_customer_id=customer.id,
        model_version_id=version.id,
        scored_by_user_id=scored_by_user_id,
        probability=probability,
        risk_band=risk_band,
        flagged=probability >= threshold,
        threshold=threshold,
        raw_input=raw_input_from_history(row),
        explanation={
            "base_value": base_value,
            "reasons": reasons,
            "source": source,
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
    customer = customer_for_queue_row(session, row)

    existing = session.scalars(
        select(Prediction)
        .where(
            Prediction.subject_customer_id == customer.id,
            Prediction.model_version_id == version.id,
        )
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).first()
    if existing is not None:
        return existing

    prediction = Prediction(
        subject_customer_id=customer.id,
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
