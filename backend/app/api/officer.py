"""The loan officer's surface: a worklist, one customer in detail, a decision."""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_explainer,
    get_scorer,
    get_session,
    live_model_version,
    user_by_email,
)
from app.db import queue_predictions
from app.db.models import Decision, User
from app.ml import artifacts, worklist
from app.schemas.decisions import DecisionCreate, DecisionOut
from app.schemas.scoring import ScoreResponse
from app.schemas.worklist import CustomerDetail, QueuePage, QueueRow

router = APIRouter(prefix="/officer", tags=["officer"])

# Reasons an officer screen shows. The engineer's /explain returns every one.
SHOWN_REASONS = 6


def _queued_or_404(customer_id: str) -> dict:
    row = worklist.find(customer_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No queued customer {customer_id}",
        )
    return row


@router.get("/queue", response_model=QueuePage)
def read_queue(
    band: str | None = Query(default=None, description="Low, Medium or High"),
    flagged: bool | None = Query(default=None),
    search: str | None = Query(default=None, description="Customer id contains"),
    minimum_months_in_arrears: int | None = Query(default=None, ge=0, le=6),
    limit: int = Query(default=worklist.DEFAULT_PAGE, ge=1, le=worklist.MAX_PAGE),
    offset: int = Query(default=0, ge=0),
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
    page = selected[offset : offset + limit]

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
    """Find a customer the way an officer would be given one: by name.

    Names live in the database and the queue lives in an artifact, so this
    searches people first and falls back to the account number. Only cardholders
    the system has met have a name, so an account nobody has opened is still
    reachable by its id and nothing else.
    """
    needle = q.strip().lower()

    named = (
        session.query(User)
        .filter(User.role == "cardholder")
        .filter(
            User.full_name.ilike(f"%{needle}%")
            | User.email.ilike(f"%{needle}%")
            | User.display_name.ilike(f"%{needle}%")
        )
        .limit(limit)
        .all()
    )
    found = {user.display_name: user.full_name for user in named}

    # Account numbers, for everyone the system has not met. Punctuation and the
    # C prefix are optional, because nobody types an id the way it is stored.
    bare = needle.upper().removeprefix("C-").removeprefix("C")
    if bare.isdigit():
        for row in worklist.filter_rows(worklist.all_rows(), search=bare)[:limit]:
            found.setdefault(row["customer_id"], None)

    return [
        {
            "customer_id": customer_id,
            "full_name": full_name,
            "known": full_name is not None,
        }
        for customer_id, full_name in list(found.items())[:limit]
    ]


@router.get("/queue/{customer_id}", response_model=QueueRow)
def read_queued_customer(customer_id: str):
    """One queue row. The shape the list uses; see /officer/customers for a page."""
    return _queued_or_404(customer_id)


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
    detail = worklist.customer_detail(_queued_or_404(customer_id))

    cardholder = (
        session.query(User)
        .filter_by(display_name=customer_id, role="cardholder")
        .one_or_none()
    )
    detail["full_name"] = cardholder.full_name if cardholder else None

    stored = queue_predictions.latest_for_cardholder(session, customer_id)
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
    row = _queued_or_404(customer_id)
    user_by_email(session, scored_by, "officer")
    version = live_model_version(session)

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
    row = _queued_or_404(customer_id)
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
    session.flush()

    return DecisionOut(**decision.to_dict())


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
    _queued_or_404(customer_id)
    cardholder = (
        session.query(queue_predictions.User)
        .filter_by(display_name=customer_id, role="cardholder")
        .one_or_none()
    )
    if cardholder is None:
        return []

    decisions = [
        decision
        for prediction in cardholder.predictions
        for decision in prediction.decisions
    ]
    return [
        DecisionOut(**decision.to_dict())
        for decision in sorted(decisions, key=lambda row: row.decided_at)
    ]
