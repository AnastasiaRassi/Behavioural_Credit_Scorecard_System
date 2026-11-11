"""The loan officer's surface.

The worklist and customer search, one customer in detail, on-demand predictions,
decisions, the shared thread and review requests, cohorts (explicit or proposed by
Loana's triage), and the per-customer officer chat.

The worklist is a precomputed artifact; a customer's score is only stored in the
database once an officer predicts or decides on it.
"""
import uuid
from dataclasses import asdict

from app.ai import officer_assistant, triage_assistant
from app.api.cardholders import account_for
from app.core import audit
from app.api.dependencies import (
    get_explainer,
    get_scorer,
    get_session,
    live_model_version,
    queued_account,
    user_by_email,
)
from app.db import chat_log, customer_registry, queue_predictions, reviews
from app.db.models import Customer, Decision, User
from app.ml import artifacts, worklist
from app.schemas.decisions import DecisionCreate, DecisionOut
from app.schemas.reviews import ReviewRequestOut, ThreadEntry
from app.schemas.scoring import ScoreResponse
from app.schemas.worklist import (
    BulkPredictRequest,
    ChatReply,
    ChatRequest,
    CohortPage,
    CustomerDetail,
    QueuePage,
    QueueRow,
    TriageFilter,
    TriageReply,
    TriageRequest,
)
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case
from sqlalchemy.orm import Session

router = APIRouter(prefix="/officer", tags=["officer"])

# Reasons an officer screen shows. The engineer's /explain returns every one.
SHOWN_REASONS = 6


def _with_customer_ids(session: Session, rows: list[dict]) -> list[dict]:
    """Give each queue row its customer's UUID, which only the database knows."""
    found = customer_registry.customers_by_account_number(
        session, [row["account_number"] for row in rows]
    )
    missing = [row["account_number"] for row in rows if row["account_number"] not in found]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Queued account {missing[0]} is not in the database. "
                "Run: python -m app.db.ingest"
            ),
        )
    return [
        {
            **row,
            "customer_id": str(found[row["account_number"]].id),
            "full_name": found[row["account_number"]].user.display_name,
        }
        for row in rows
    ]


@router.get("/queue", response_model=QueuePage)
def read_queue(
    band: str | None = Query(default=None, description="Low, Medium or High"),
    flagged: bool | None = Query(default=None),
    search: str | None = Query(default=None, description="Account number contains"),
    minimum_months_in_arrears: int | None = Query(default=None, ge=0, le=6),
    limit: int = Query(default=worklist.DEFAULT_PAGE, ge=1, le=worklist.MAX_PAGE),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    """A page of the worklist, riskiest first.

    Paged because the queue is every held-out customer. Both the filtered and the
    unfiltered summaries are returned so a screen can say "41 of 4,500" without
    asking twice.
    """
    if band is not None and band not in worklist.BANDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"band must be one of {', '.join(worklist.BANDS)}, got {band!r}",
        )

    everything = worklist.all_rows()
    selected = worklist.filter_rows(
        everything,
        band=band,
        flagged=flagged,
        search=search,
        minimum_months_in_arrears=minimum_months_in_arrears,
    )
    page = _with_customer_ids(session, selected[offset : offset + limit])

    return QueuePage(
        total=len(selected),
        offset=offset,
        returned=len(page),
        threshold=artifacts.scoring_config()["threshold_f1_optimal"],
        summary=worklist.summarise(selected),
        summary_all=worklist.summarise(everything),
        rows=page,
    )


@router.get("/customers", response_model=list[dict])
def search_customers(
    q: str = Query(min_length=1, description="Name, email or account number"),
    limit: int = Query(default=10, ge=1, le=50),
    session: Session = Depends(get_session),
):
    """Find a customer the way an officer would be given one: by name or number.

    Every customer has a name, so this matches the name, the account number or,
    for those who registered, the email. Only customers in the worklist are
    returned: the others have no queue row to open. An exact account number is
    listed first, then customers who can sign in.
    """
    needle = q.strip()
    queued = [row["account_number"] for row in worklist.all_rows()]

    matches = (
        session.query(Customer, User)
        .join(User, Customer.user_id == User.id)
        .filter(
            Customer.account_number.in_(queued),
            User.display_name.icontains(needle, autoescape=True)
            | User.email.icontains(needle, autoescape=True)
            | Customer.account_number.icontains(needle, autoescape=True),
        )
        .order_by(
            case((Customer.account_number == needle, 0), else_=1),
            # Only 144 invented names exist, so a name matches dozens of people;
            # those with a login are the ones anyone is likely to be looking for.
            case((User.password_hash.is_not(None), 0), else_=1),
            User.display_name,
            Customer.account_number,
        )
        .limit(limit)
        .all()
    )
    # The search text is a name or email, so only its length and the result count are logged.
    audit.event("customer_search", query_length=len(needle), results=len(matches))
    return [
        {
            "customer_id": str(customer.id),
            "account_number": customer.account_number,
            "full_name": user.display_name,
            "known": user.can_sign_in,
        }
        for customer, user in matches
    ]


@router.get("/review-requests", response_model=list[ReviewRequestOut])
def read_review_requests(session: Session = Depends(get_session)):
    """Customers waiting for a person to look at their account, oldest first."""
    open_requests = reviews.all_open(session)
    audit.event("inbox_viewed", open=len(open_requests))
    return [
        reviews.to_dict(request, customer, user)
        for request, customer, user in open_requests
    ]


@router.get("/queue/{customer_id}", response_model=QueueRow)
def read_queued_customer(customer_id: str, session: Session = Depends(get_session)):
    """One queue row, in the shape the list uses. /officer/customers/{id} is the full page."""
    return queued_account(session, customer_id)[1]


@router.get("/customers/{customer_id}", response_model=CustomerDetail)
def read_customer_detail(
    customer_id: str, session: Session = Depends(get_session)
):
    """Everything a dedicated customer page needs.

    The queue row plus the six-month history, the behavioural segment and the
    highlights derived from that history. A queue row is right for a list and too
    thin for a page.

    `latest_prediction` is the most recent score an officer ran on this customer,
    or None if nobody has. It is reported separately from the queue score rather
    than replacing it, so the screen can show that a fresh prediction agrees with
    the overnight one instead of silently overwriting it.
    """
    customer, row = queued_account(session, customer_id)
    audit.event("customer_viewed", customer=customer_id, account=customer.account_number)
    detail = worklist.customer_detail(row)

    detail["full_name"] = customer.user.display_name

    stored = queue_predictions.latest_for_cardholder(session, customer.id)
    detail["latest_prediction"] = stored.to_dict() if stored else None
    return detail


@router.post(
    "/customers/{customer_id}/predict",
    response_model=ScoreResponse,
    status_code=status.HTTP_201_CREATED,
)
def predict_default(
    customer_id: str,
    scored_by: str = Query(description="Email of the officer running the prediction"),
    session: Session = Depends(get_session),
):
    """Run the model on this customer now, store the score, return it explained.

    This is the officer's "predict default" action. It scores through the live
    model rather than reading the queue, so the officer sees a number produced
    on the spot and the reasons behind it, and the score is recorded before any
    decision points at it.

    The inputs are the nineteen columns the queue carries. The four protected
    attributes are not among them and are not needed: they never reach the model.
    """
    officer = user_by_email(session, scored_by, "officer")
    return _predict(session, customer_id, live_model_version(session), officer)


def _predict(
    session: Session, customer_id: str, version, officer: User
) -> ScoreResponse:
    """Score one worklist customer through the live model and store the result."""
    _, row = queued_account(session, customer_id)

    customer = queue_predictions.scored_columns(row)
    result = get_scorer().score_one(customer)
    reasons, base_value = get_explainer().reasons(customer, limit=SHOWN_REASONS)
    reason_dicts = [
        {**reason.to_dict(), "raises_risk": reason.raises_risk} for reason in reasons
    ]

    prediction = queue_predictions.store_fresh_prediction(
        session,
        row,
        version,
        probability=result.probability,
        risk_band=result.risk_band,
        threshold=result.threshold,
        reasons=reason_dicts,
        base_value=base_value,
        scored_by_user_id=officer.id,
    )

    audit.event(
        "predict",
        actor=officer.email,
        customer=customer_id,
        account=row["account_number"],
        prediction=prediction.id,
        band=result.risk_band,
        flagged=result.flagged,
        model=version.version,
    )
    return ScoreResponse(
        prediction_id=str(prediction.id),
        probability=result.probability,
        risk_band=result.risk_band,
        threshold=result.threshold,
        flagged=result.flagged,
        band_default_rate=result.band_default_rate,
        model_version=version.version,
        reasons=reason_dicts,
    )


@router.post(
    "/predictions", response_model=list[ScoreResponse], status_code=status.HTTP_201_CREATED
)
def predict_selected(request: BulkPredictRequest, session: Session = Depends(get_session)):
    """The officer's "predict selected": the single predict, once per ticked customer.

    Not batch scoring: capped at ten, every score is stored against the officer who
    asked for it, and any unknown customer fails the whole request.
    """
    officer = user_by_email(session, request.scored_by, "officer")
    version = live_model_version(session)
    audit.event(
        "predict_selected",
        actor=request.scored_by,
        count=len(request.customer_ids),
    )
    return [
        _predict(session, customer_id, version, officer)
        for customer_id in request.customer_ids
    ]


def _cohort(session: Session, criteria: TriageFilter) -> CohortPage:
    """The customers a filter selects, with their latest officer-run score if any."""
    dumped = criteria.model_dump()
    matching = worklist.select_cohort(
        worklist.all_rows(), dumped, queue_predictions.scored_account_numbers(session)
    )
    page = _with_customer_ids(session, matching[: criteria.limit])
    latest = queue_predictions.latest_by_customer(
        session, [uuid.UUID(row["customer_id"]) for row in page]
    )
    rows = []
    for row in page:
        scored = latest.get(uuid.UUID(row["customer_id"]))
        rows.append(
            {
                **row,
                "last_risk_band": scored.risk_band if scored else None,
                "last_scored_at": scored.created_at.isoformat() if scored else None,
            }
        )
    return CohortPage(
        description=worklist.describe_cohort(dumped), total=len(matching), rows=rows
    )


@router.post("/cohort", response_model=CohortPage)
def read_cohort(criteria: TriageFilter, session: Session = Depends(get_session)):
    """A cohort from an explicit filter: the quick cohorts, or a filter Loana proposed."""
    page = _cohort(session, criteria)
    audit.event("cohort", filter=page.description, total=page.total, shown=len(page.rows))
    return page


@router.post("/triage", response_model=TriageReply)
def triage(request: TriageRequest, session: Session = Depends(get_session)):
    """Loana's triage turn: the officer's request becomes a filter, then a cohort.

    Not written to the chat log yet: chat_turn rows belong to one customer, and a
    triage request is about none.
    """
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="messages must end with the officer's request",
        )

    turn = triage_assistant.interpret([m.model_dump() for m in request.messages])
    audit.event(
        "triage",
        turns=len(request.messages),
        source=turn.source,
        filter_proposed=turn.filter is not None,
    )
    return TriageReply(
        reply=turn.reply,
        source=turn.source,
        note=turn.note,
        filter=turn.filter,
        cohort=_cohort(session, turn.filter) if turn.filter else None,
    )


@router.post(
    "/customers/{customer_id}/decisions",
    response_model=DecisionOut,
    status_code=status.HTTP_201_CREATED,
)
def decide_on_queued_customer(
    customer_id: str,
    request: DecisionCreate,
    session: Session = Depends(get_session),
):
    """Record an intervention on a queued customer.

    The queue is an artifact and decisions point at a prediction row, so the
    customer's score is materialised into the database the first time anyone acts
    on it, then reused. The stored score is the one from the queue: it is the
    number the officer was looking at, and the blind feature set the queue is
    built from could not be re-scored anyway.
    """
    customer, row = queued_account(session, customer_id)
    officer = user_by_email(session, request.decided_by, "officer")
    version = live_model_version(session)

    prediction = queue_predictions.prediction_for(
        session, row, version, artifacts.scoring_config()["threshold_f1_optimal"]
    )

    decision = Decision(
        prediction_id=prediction.id,
        decided_by_user_id=officer.id,
        action=request.action,
        justification=request.justification.strip(),
    )
    session.add(decision)
    # The decision is the officer's answer, so it closes the conversation.
    closed = reviews.resolve_open(session, customer.id)
    session.flush()

    # The justification is deliberately not logged: it is free text about a person.
    audit.event(
        "decision",
        actor=request.decided_by,
        customer=customer_id,
        account=customer.account_number,
        decision=decision.id,
        prediction=prediction.id,
        decided_action=request.action,
        band=prediction.risk_band,
        overrode_model=decision.to_dict()["overrode_model"],
        messages_closed=closed,
    )
    return DecisionOut(**decision.to_dict())


@router.get("/customers/{customer_id}/thread", response_model=list[ThreadEntry])
def read_customer_thread(customer_id: str, session: Session = Depends(get_session)):
    """The same thread the cardholder reads: their messages and every decision."""
    customer, _ = queued_account(session, customer_id)
    return reviews.thread(session, customer)


@router.get(
    "/customers/{customer_id}/decisions", response_model=list[DecisionOut]
)
def read_customer_decisions(
    customer_id: str, session: Session = Depends(get_session)
):
    """The decision trail for a queued customer, oldest first.

    Empty until someone acts: a queued customer has no prediction row until the
    first decision materialises one.
    """
    customer, _ = queued_account(session, customer_id)

    decisions = [
        decision
        for prediction in customer.predictions
        for decision in prediction.decisions
    ]
    return [
        DecisionOut(**decision.to_dict())
        for decision in sorted(decisions, key=lambda row: row.decided_at)
    ]


@router.post("/customers/{customer_id}/chat", response_model=ChatReply)
def chat_about_customer(
    customer_id: str, request: ChatRequest, session: Session = Depends(get_session)
):
    """The officer assistant's next turn, answered from this customer's own facts.

    Stateless like the cardholder chat: the client sends the conversation so far,
    which must end with the officer's question.
    """
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="messages must end with the officer's question",
        )

    row, stored, record = account_for(customer_id, session)
    audit.event("officer_chat", customer=customer_id, turns=len(request.messages))
    return officer_assistant.answer(
        row,
        [message.model_dump() for message in request.messages],
        stored_prediction=stored,
        record=record,
        on_turn=lambda turn: chat_log.record(session, asdict(turn)),
    ).to_dict()
