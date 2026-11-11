"""Role and user listings for the landing and sign-in screens.

Read-only and unauthenticated: they say which roles exist and which accounts can
sign in, nothing more. Proving who you are is `sessions.py`.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.db.models import ROLES, User
from app.schemas.worklist import RoleOut

router = APIRouter(tags=["roles"])

DESCRIPTIONS = {
    "engineer": (
        "Model metrics, fairness audit, threshold economics, feature importance "
        "and drift.",
        "/report",
    ),
    "officer": (
        "A worklist of scored cardholders, with the reasons behind each score "
        "and a decision to record.",
        "/officer/queue",
    ),
    "cardholder": (
        "Your own account: whether your next payment looks at risk, and what "
        "would help most.",
        "/cardholders/{customer_id}",
    ),
}

LABELS = {
    "engineer": "ML engineer",
    "officer": "Loan officer",
    "cardholder": "Cardholder",
}


@router.get("/roles", response_model=list[RoleOut])
def read_roles(session: Session = Depends(get_session)):
    """The three roles, with where each one starts.

    Driven by the database's own role vocabulary, so a role cannot appear here
    that the schema would reject.
    """
    counts = {role: 0 for role in ROLES}
    # People who can sign in. Every customer is a user, so counting them all would
    # report thousands of cardholders nobody has registered.
    for role, total in (
        session.query(User.role, func.count(User.id))
        .filter(User.password_hash.is_not(None))
        .group_by(User.role)
        .all()
    ):
        counts[role] = total

    return [
        RoleOut(
            role=role,
            label=LABELS[role],
            description=DESCRIPTIONS[role][0],
            landing_endpoint=DESCRIPTIONS[role][1],
            users=counts.get(role, 0),
        )
        for role in ROLES
    ]


def _listed(user: User) -> dict:
    """A user as a picker needs them, plus which customer a cardholder is."""
    # A user id is not a customer id, so a screen that has only picked a
    # cardholder still cannot address their account. None for staff, who are not
    # customers at all.
    registered_as = user.customer
    return {
        **user.to_dict(),
        "customer_id": str(registered_as.id) if registered_as else None,
        "account_number": registered_as.account_number if registered_as else None,
    }


@router.get("/users")
def read_users(
    role: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[dict]:
    """Known users, so a screen can offer "who are you" after a role is picked.

    Cardholders are omitted unless asked for by role, and then only the ones who
    can sign in: the worklist is the way in to a customer, and listing every
    scored person would be a directory of people at risk.
    """
    query = session.query(User).filter(User.password_hash.is_not(None))
    if role:
        query = query.filter_by(role=role)
    else:
        query = query.filter(User.role != "cardholder")

    return [
        _listed(user)
        for user in query.order_by(User.role, User.display_name).all()
    ]
