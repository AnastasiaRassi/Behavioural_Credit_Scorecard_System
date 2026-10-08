"""Scoring endpoints. Routing and persistence only; the maths lives in `ml/`."""
import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_explainer,
    get_scorer,
    get_session,
    live_model_version,
    user_by_email,
)
from app.db.models import Prediction, User
from app.ml.feature_engineering import RAW_COLUMNS
from app.schemas.scoring import (
    BatchRow,
    BatchScoreResponse,
    ScoreRequest,
    ScoreResponse,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])

# How many reasons a screen shows. The full set is every feature with a non
# negligible effect, which is more than anyone reads.
SHOWN_REASONS = 6

# A batch larger than this is a job, not a request.
MAX_BATCH_ROWS = 5000


def _cardholder(session: Session, reference: str | None) -> User:
    """Find the named cardholder, or create one for a walk-in."""
    if reference:
        existing = session.query(User).filter_by(
            display_name=reference, role="cardholder"
        ).one_or_none()
        if existing is not None:
            return existing

    cardholder = User(
        display_name=reference or "Unnamed cardholder", role="cardholder"
    )
    session.add(cardholder)
    session.flush()
    return cardholder


@router.post("", response_model=ScoreResponse, status_code=status.HTTP_201_CREATED)
def score_customer(
    request: ScoreRequest,
    session: Session = Depends(get_session),
):
    """Score one cardholder, store the result with its explanation, return both.

    The explanation is stored rather than recomputed later: `train.py` prunes
    artifacts/, so a past model's binary does not survive a retrain and the
    reasons shown today could not be reproduced afterwards.
    """
    scorer = get_scorer()
    explainer = get_explainer()
    version = live_model_version(session)

    customer = request.customer.model_dump()
    result = scorer.score_one(customer)
    reasons, base_value = explainer.reasons(customer, limit=SHOWN_REASONS)

    if request.scored_by:
        user_by_email(session, request.scored_by, "officer")

    prediction = Prediction(
        subject_user_id=_cardholder(session, request.cardholder_reference).id,
        model_version_id=version.id,
        probability=result.probability,
        risk_band=result.risk_band,
        flagged=result.flagged,
        threshold=result.threshold,
        raw_input=customer,
        explanation={
            "base_value": base_value,
            "reasons": [reason.to_dict() for reason in reasons],
        },
    )
    session.add(prediction)
    session.flush()

    return ScoreResponse(
        prediction_id=str(prediction.id),
        probability=result.probability,
        risk_band=result.risk_band,
        threshold=result.threshold,
        flagged=result.flagged,
        band_default_rate=result.band_default_rate,
        model_version=version.version,
        reasons=[
            {**reason.to_dict(), "raises_risk": reason.raises_risk}
            for reason in reasons
        ],
    )


@router.get("/{prediction_id}", response_model=dict)
def read_prediction(prediction_id: str, session: Session = Depends(get_session)):
    """A stored score with the reasons as they were shown at the time."""
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction {prediction_id}",
        )

    stored = prediction.to_dict()
    stored["model_version"] = prediction.model_version.version
    stored["cardholder"] = prediction.subject.display_name
    stored["decisions"] = [decision.to_dict() for decision in prediction.decisions]
    return stored


@router.post("/batch", response_model=BatchScoreResponse)
async def score_csv(file: UploadFile):
    """Score a CSV of cardholders.

    Deliberately does not persist. A batch is a bulk review, not a set of
    decisions, and writing thousands of unactioned rows would bury the real
    decision trail. Rows are reported with their original position so a rejected
    row can be found in the uploaded file.
    """
    raw = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Could not read {file.filename} as CSV: {exception}",
        ) from exception

    if len(frame) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{len(frame)} rows exceeds the {MAX_BATCH_ROWS} row limit",
        )

    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Missing required columns: {missing}",
        )

    scorer = get_scorer()
    rows: list[BatchRow] = []
    scored = 0

    # Scored one row at a time so a single bad row does not reject the file.
    for position, (_, record) in enumerate(frame.iterrows(), start=1):
        try:
            result = scorer.score_one(record[RAW_COLUMNS].to_dict())
        except Exception as exception:
            rows.append(BatchRow(row=position, error=str(exception)))
            continue
        scored += 1
        rows.append(
            BatchRow(
                row=position,
                probability=result.probability,
                risk_band=result.risk_band,
                flagged=result.flagged,
            )
        )

    return BatchScoreResponse(
        scored=scored, rejected=len(rows) - scored, rows=rows
    )
