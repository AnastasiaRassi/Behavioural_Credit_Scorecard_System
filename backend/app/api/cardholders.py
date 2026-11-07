"""The cardholder's own view of their account.

A behavioural scorecard re-scores someone who already holds the card, so this is
an early warning, never a decision. The payload reflects that: it carries the
band and the reasons, not the probability or the threshold.

Who may read it is decided by `dependencies.account_reader`, not here: narrowing
the payload is pointless if anyone can ask for anyone's.
"""
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.ai import assistant
from app.api.dependencies import account_reader, get_session
from app.db import chat_log, queue_predictions, statements
from app.ml import worklist
from app.schemas.worklist import CardholderView, ChatReply, ChatRequest

router = APIRouter(prefix="/cardholders", tags=["cardholder"])


@router.get(
    "/{customer_id}",
    response_model=CardholderView,
    dependencies=[Depends(account_reader)],
)
def read_ownaccount_for(
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
    row, stored, record = account_for(customer_id, session)
    return worklist.cardholder_view(row, stored_prediction=stored, record=record)


@router.post(
    "/{customer_id}/chat",
    response_model=ChatReply,
    dependencies=[Depends(account_reader)],
)
def chat_aboutaccount_for(
    customer_id: str, request: ChatRequest, session: Session = Depends(get_session)
):
    """The assistant's next turn, answered from this account's own facts.

    Stateless: the client sends the conversation so far. An empty conversation
    returns Loana's opening interpretation of the account.
    """
    row, stored, record = account_for(customer_id, session)
    messages = [message.model_dump() for message in request.messages]
    return assistant.answer(
        row,
        messages,
        stored_prediction=stored,
        record=record,
        on_turn=lambda turn: chat_log.record(session, asdict(turn)),
    ).to_dict()


def account_for(customer_id: str, session: Session):
    """The queue row, the latest officer score if any, and the raw input for the search."""
    row = worklist.find(customer_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account {customer_id}",
        )

    stored = queue_predictions.latest_for_cardholder(session, customer_id)
    try:
        record = statements.raw_input_for(session, customer_id)
    except statements.CustomerNotFound:
        record = None
    return row, stored, record
