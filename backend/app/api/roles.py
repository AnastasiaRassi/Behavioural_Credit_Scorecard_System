"""The landing page: pick a role, then pick who you are.

Role selection is not authentication. It decides which screens open and which
payload shape the API returns, and the system says so rather than implying a
login took place.
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
    for role, total in (
        session.query(User.role, func.count(User.id)).group_by(User.role).all()
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


@router.get("/users")
def read_users(
    role: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[dict]:
    """Known users, so a screen can offer "who are you" after a role is picked.

    Cardholders are omitted unless asked for by name: the worklist is the way in
    to a customer, and listing every scored person would be a directory of
    people at risk.
    """
    query = session.query(User)
    if role:
        query = query.filter_by(role=role)
    else:
        query = query.filter(User.role != "cardholder")

    return [
        {
            "id": str(user.id),
            "display_name": user.display_name,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role,
        }
        for user in query.order_by(User.role, User.display_name).all()
    ]
