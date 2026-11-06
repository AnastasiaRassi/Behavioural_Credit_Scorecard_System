"""Recourse: the smallest change that would move an account to a better band.

Explanations say why a score is what it is. This says what to do about it,
which is the question a flagged customer actually asks, and the one a right to
explanation is thin without.

Guarded by `account_reader`, like the cardholder's own view: a cardholder
reaches only the account they registered as, staff reach any. The answer is
built to be safe for either, so there is one endpoint rather than two — bands
and actions, no probability.

The maths is in `ml/counterfactuals.py` and the wording is in
`ml/explainability.py`; this module only routes between them (conventions §1).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import account_reader, get_recourse_explainer, get_session
from app.db import statements
from app.ml.counterfactuals import NEXT_BAND_UP, CounterfactualExplainer, RecourseTarget
from app.ml.explainability import describe_change
from app.schemas.recourse import RecourseStep, RecourseView

router = APIRouter(prefix="/cardholders", tags=["cardholder"])


@router.get(
    "/{customer_id}/recourse",
    response_model=RecourseView,
    dependencies=[Depends(account_reader)],
)
def read_recourse(
    customer_id: str,
    session: Session = Depends(get_session),
    explainer: CounterfactualExplainer = Depends(get_recourse_explainer),
):
    """What would move this account one band better, in plain language."""
    try:
        record = statements.raw_input_for(session, customer_id)
    except statements.CustomerNotFound as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exception)
        ) from exception

    current_band = explainer.scorer.score_one(record).risk_band
    target_band = NEXT_BAND_UP.get(current_band)
    if target_band is None:
        # Already in the best band: there is nothing to ask of this customer,
        # and saying so is the honest answer rather than an empty result.
        return RecourseView(
            customer_id=customer_id,
            current_band=current_band,
            target_band=current_band,
            reachable=True,
            achieved_band=current_band,
            note="This account is already in the lowest risk band.",
        )

    result = explainer.explain(record, RecourseTarget(band=target_band))

    steps = [
        RecourseStep(
            description=describe_change(feature, change["from"], change["to"]),
            feature=feature,
            before=change["from"],
            after=change["to"],
        )
        for feature, change in result.feature_changes.items()
    ]

    return RecourseView(
        customer_id=customer_id,
        current_band=current_band,
        target_band=target_band,
        reachable=result.is_feasible,
        achieved_band=result.achieved_band,
        steps=steps,
        note=_note(result.is_feasible, target_band, steps),
    )


def _note(reachable: bool, target_band: str, steps: list[RecourseStep]) -> str:
    if reachable and steps:
        return f"Doing all of this would move the account to {target_band} risk."
    if reachable:
        return "No change needed to reach the next band."
    return (
        "No change within this account's reach would get it to "
        f"{target_band} risk on its own. These steps are the furthest the "
        "search got."
    )
