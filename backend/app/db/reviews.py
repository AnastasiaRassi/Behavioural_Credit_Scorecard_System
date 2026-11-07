"""Cardholder messages (`review_request`) and the account's shared activity thread.

Each `review_request` row is one message from the cardholder. Unresolved rows are
the officer's inbox; an officer decision resolves every open row for that customer,
which closes the conversation until the cardholder writes again.

The thread is derived from the messages and the decisions rather than stored a
second time, so the officer's audit trail and what the cardholder reads cannot
disagree.
"""
import uuid

from sqlalchemy.orm import Session

from app.db.models import (
    OPEN_REVIEW_STATUSES,
    Customer,
    Decision,
    Prediction,
    ReviewRequest,
    User,
)


def open_for(session: Session, customer_id: uuid.UUID) -> ReviewRequest | None:
    """The customer's oldest message still waiting for a person."""
    return (
        session.query(ReviewRequest)
        .filter(
            ReviewRequest.customer_id == customer_id,
            ReviewRequest.status.in_(OPEN_REVIEW_STATUSES),
        )
        .order_by(ReviewRequest.requested_at)
        .first()
    )


def open_request(session: Session, customer_id: uuid.UUID, reason: str) -> ReviewRequest:
    """Add a cardholder message to the thread; it stays open until an officer decides."""
    request = ReviewRequest(customer_id=customer_id, reason=reason.strip())
    session.add(request)
    session.flush()
    return request


def resolve_open(session: Session, customer_id: uuid.UUID) -> int:
    """Close the conversation: every open message is answered by the decision just taken."""
    return (
        session.query(ReviewRequest)
        .filter(
            ReviewRequest.customer_id == customer_id,
            ReviewRequest.status.in_(OPEN_REVIEW_STATUSES),
        )
        .update({ReviewRequest.status: "resolved"}, synchronize_session=False)
    )


def all_open(session: Session) -> list[tuple[ReviewRequest, Customer, User]]:
    return (
        session.query(ReviewRequest, Customer, User)
        .join(Customer, ReviewRequest.customer_id == Customer.id)
        .join(User, Customer.user_id == User.id)
        .filter(ReviewRequest.status.in_(OPEN_REVIEW_STATUSES))
        .order_by(ReviewRequest.requested_at)
        .all()
    )


def thread(session: Session, customer: Customer) -> list[dict]:
    """Messages and decisions for one account, oldest first."""
    messages = (
        session.query(ReviewRequest)
        .filter(ReviewRequest.customer_id == customer.id)
        .all()
    )
    decisions = (
        session.query(Decision, User)
        .join(Prediction, Decision.prediction_id == Prediction.id)
        .join(User, Decision.decided_by_user_id == User.id)
        .filter(Prediction.subject_customer_id == customer.id)
        .all()
    )
    entries = [
        {
            "id": str(message.id),
            "kind": "message",
            "at": message.requested_at,
            "author": customer.user.display_name or "Cardholder",
            "body": message.reason,
            "action": None,
            "status": message.status,
        }
        for message in messages
    ] + [
        {
            "id": str(decision.id),
            "kind": "decision",
            "at": decision.decided_at,
            "author": officer.display_name or "Loan officer",
            "body": decision.justification,
            "action": decision.action,
            "status": None,
        }
        for decision, officer in decisions
    ]
    entries.sort(key=lambda entry: entry["at"])
    return [{**entry, "at": entry["at"].isoformat()} for entry in entries]


def to_dict(request: ReviewRequest, customer: Customer, user: User) -> dict:
    return {
        "request_id": str(request.id),
        "customer_id": str(customer.id),
        "account_number": customer.account_number,
        "full_name": user.display_name,
        "requested_at": request.requested_at.isoformat(),
        "reason": request.reason,
        "status": request.status,
    }
