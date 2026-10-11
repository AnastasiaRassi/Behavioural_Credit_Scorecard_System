"""The cardholder's own view of their account.

A behavioural scorecard re-scores someone who already holds the card, so this is
an early warning, never a decision. The payload reflects that: it carries the
band and the reasons, not the probability or the threshold.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.db import queue_predictions
from app.ml import worklist
from app.schemas.worklist import CardholderView

router = APIRouter(prefix="/cardholders", tags=["cardholder"])


@router.get("/{customer_id}", response_model=CardholderView)
def read_own_account(
    customer_id: str, session: Session = Depends(get_session)
):
    """What this cardholder is told about their next payment.

    The same underlying score the officer sees, reduced to what the cardholder is
    owed: the band, whether the account needs attention, and the drivers in plain
    language with their direction.

    Prefers the most recent score an officer has run, so the customer and the
    officer are looking at the same prediction rather than the customer seeing a
    stale overnight one. Falls back to the queue when none has been run.
    """
    row = worklist.find(customer_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account {customer_id}",
        )

    stored = queue_predictions.latest_for_cardholder(session, customer_id)
    return worklist.cardholder_view(row, stored_prediction=stored)
