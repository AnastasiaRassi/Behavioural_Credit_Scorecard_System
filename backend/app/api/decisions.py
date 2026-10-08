"""Decision endpoints: what an officer did about a score, and why."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, user_by_email
from app.db.models import Decision, Prediction
from app.schemas.decisions import DecisionCreate, DecisionOut

router = APIRouter(tags=["decisions"])


def _prediction_or_404(session: Session, prediction_id: str) -> Prediction:
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction {prediction_id}",
        )
    return prediction


@router.post(
    "/predictions/{prediction_id}/decisions",
    response_model=DecisionOut,
    status_code=status.HTTP_201_CREATED,
)
def record_decision(
    prediction_id: str,
    request: DecisionCreate,
    session: Session = Depends(get_session),
):
    """Record an intervention against a score.

    A prediction can be acted on more than once, so this appends rather than
    replaces: an account monitored last week and escalated today keeps both
    entries, which is the point of an audit trail.
    """
    prediction = _prediction_or_404(session, prediction_id)
    officer = user_by_email(session, request.decided_by, "officer")

    decision = Decision(
        prediction_id=prediction.id,
        decided_by_user_id=officer.id,
        action=request.action,
        justification=request.justification.strip(),
    )
    session.add(decision)
    session.flush()

    return DecisionOut(**decision.to_dict())


@router.get("/predictions/{prediction_id}/decisions", response_model=list[DecisionOut])
def list_decisions(prediction_id: str, session: Session = Depends(get_session)):
    """Every decision taken on one score, oldest first."""
    prediction = _prediction_or_404(session, prediction_id)
    return [
        DecisionOut(**decision.to_dict())
        for decision in sorted(prediction.decisions, key=lambda d: d.decided_at)
    ]


@router.get("/decisions", response_model=list[DecisionOut])
def recent_decisions(limit: int = 50, session: Session = Depends(get_session)):
    """The decision trail, newest first."""
    rows = (
        session.query(Decision)
        .order_by(Decision.decided_at.desc())
        .limit(min(limit, 500))
        .all()
    )
    return [DecisionOut(**decision.to_dict()) for decision in rows]
