"""Cardholder-facing endpoints: own account view, Loana chat, and messages to the officer.

The score is an early warning, never a decision, so the view carries the band and
reasons but not the probability or threshold (see `CardholderView`).

Every route is guarded by `dependencies.account_reader`, not here: narrowing the
payload is pointless if anyone can ask for anyone's account.
"""
from dataclasses import asdict

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.ai import assistant
from app.core import audit
from app.api.dependencies import (
    account_reader,
    customer_or_404,
    get_explainer,
    get_scorer,
    get_session,
    live_model_version,
    queued_account,
)
from app.db import chat_log, queue_predictions, reviews, statements
from app.ml import worklist
from app.schemas.reviews import ReviewRequestCreate, ReviewRequestOut, ThreadEntry
from app.schemas.worklist import CardholderView, ChatReply, ChatRequest

router = APIRouter(prefix="/cardholders", tags=["cardholder"])

SHOWN_REASONS = 6  # matches the officer's predict


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
    audit.event("cardholder_chat", customer=customer_id, turns=len(messages))
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
    # Only the length is logged: the message is the cardholder's own words.
    audit.event(
        "review_request",
        customer=customer_id,
        account=customer.account_number,
        request=request.id,
        message_length=len(body.reason),
    )
    return reviews.to_dict(request, customer, customer.user)


@router.get(
    "/{customer_id}/thread",
    response_model=list[ThreadEntry],
    dependencies=[Depends(account_reader)],
)
def read_thread(customer_id: str, session: Session = Depends(get_session)):
    """The account's messages and decisions, oldest first: what the officer sees too."""
    return reviews.thread(session, customer_or_404(session, customer_id))


def _score_live(session: Session, row: dict):
    """Score through the live model and store it, so the first view isn't a queue copy."""
    customer = queue_predictions.scored_columns(row)
    result = get_scorer().score_one(customer)
    reasons, base_value = get_explainer().reasons(customer, limit=SHOWN_REASONS)
    prediction = queue_predictions.store_fresh_prediction(
        session,
        row,
        live_model_version(session),
        probability=result.probability,
        risk_band=result.risk_band,
        threshold=result.threshold,
        reasons=[{**r.to_dict(), "raises_risk": r.raises_risk} for r in reasons],
        base_value=base_value,
        scored_by_user_id=None,
        source="cardholder_view",
    )
    audit.event(
        "predict",
        actor="cardholder",
        account=row["account_number"],
        prediction=prediction.id,
        band=result.risk_band,
    )
    return prediction


def account_for(customer_id: str, session: Session):
    """The queue row, the latest officer score if any, and the raw input for the search."""
    customer, row = queued_account(session, customer_id)

    stored = queue_predictions.latest_for_cardholder(session, customer.id)
    if stored is None:
        stored = _score_live(session, row)
    try:
        record = statements.raw_input_for(session, customer.id)
    except statements.CustomerNotFound:
        record = None
    return row, stored, record
