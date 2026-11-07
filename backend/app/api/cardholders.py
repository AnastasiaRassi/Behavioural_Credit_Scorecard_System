"""The cardholder's own view of their account.

A behavioural scorecard re-scores someone who already holds the card, so this is
an early warning, never a decision. The payload reflects that: it carries the
band and the reasons, not the probability or the threshold.

Who may read it is decided by `dependencies.account_reader`, not here: narrowing
the payload is pointless if anyone can ask for anyone's.
"""
from dataclasses import asdict

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.ai import assistant
from app.api.dependencies import (
    account_reader,
    customer_or_404,
    get_session,
    queued_account,
)
from app.db import chat_log, queue_predictions, reviews, statements
from app.ml import worklist
from app.schemas.reviews import ReviewRequestCreate, ReviewRequestOut, ThreadEntry
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


@router.get(
    "/{customer_id}/review-request",
    response_model=ReviewRequestOut | None,
    dependencies=[Depends(account_reader)],
)
def read_open_review_request(
    customer_id: str, session: Session = Depends(get_session)
):
    """The cardholder's open request for a human review, or null."""
    customer = customer_or_404(session, customer_id)
    request = reviews.open_for(session, customer.id)
    return None if request is None else reviews.to_dict(request, customer, customer.user)


@router.post(
    "/{customer_id}/review-request",
    response_model=ReviewRequestOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(account_reader)],
)
def request_human_review(
    customer_id: str,
    body: ReviewRequestCreate,
    session: Session = Depends(get_session),
):
    """Send the loan officer a message; it reopens the conversation if it was closed."""
    customer = customer_or_404(session, customer_id)
    request = reviews.open_request(session, customer.id, body.reason)
    return reviews.to_dict(request, customer, customer.user)


@router.get(
    "/{customer_id}/thread",
    response_model=list[ThreadEntry],
    dependencies=[Depends(account_reader)],
)
def read_thread(customer_id: str, session: Session = Depends(get_session)):
    """The account's messages and decisions, oldest first: what the officer sees too."""
    return reviews.thread(session, customer_or_404(session, customer_id))


def account_for(customer_id: str, session: Session):
    """The queue row, the latest officer score if any, and the raw input for the search."""
    customer, row = queued_account(session, customer_id)

    stored = queue_predictions.latest_for_cardholder(session, customer.id)
    try:
        record = statements.raw_input_for(session, customer.id)
    except statements.CustomerNotFound:
        record = None
    return row, stored, record
