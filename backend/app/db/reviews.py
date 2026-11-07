"""Opening and listing human-review requests (`review_request`)."""
import uuid

from sqlalchemy.orm import Session

from app.db.models import OPEN_REVIEW_STATUSES, Customer, ReviewRequest, User


def open_for(session: Session, customer_id: uuid.UUID) -> ReviewRequest | None:
    """The customer's oldest request still waiting for a person."""
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
    """Record a request, or return the one already open.

    One open request per customer: a second click must not queue the same person
    twice, and the officer reads the first note in the customer's own words.
    """
    existing = open_for(session, customer_id)
    if existing is not None:
        return existing

    request = ReviewRequest(customer_id=customer_id, reason=reason.strip())
    session.add(request)
    session.flush()
    return request


def all_open(session: Session) -> list[tuple[ReviewRequest, Customer, User]]:
    return (
        session.query(ReviewRequest, Customer, User)
        .join(Customer, ReviewRequest.customer_id == Customer.id)
        .join(User, Customer.user_id == User.id)
        .filter(ReviewRequest.status.in_(OPEN_REVIEW_STATUSES))
        .order_by(ReviewRequest.requested_at)
        .all()
    )


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
