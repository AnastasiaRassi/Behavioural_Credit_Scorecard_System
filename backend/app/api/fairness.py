"""Fairness audit endpoint. Reads the audit; never recomputes it."""
from fastapi import APIRouter

from app.ml import artifacts

router = APIRouter(prefix="/fairness", tags=["fairness"])

FOUR_FIFTHS = 0.8


@router.get("")
def read_audit() -> dict:
    """The audit as evaluated at training time, with a pass or fail summary.

    Only disparate impact is reported as a conclusion. The equalized-odds gaps
    are included but marked unstable, because they move substantially between
    random splits and sometimes reverse sign, so they cannot support a claim
    about a specific group.
    """
    audit = artifacts.fairness_detail()
    breaches = [
        attribute
        for attribute, detail in audit.items()
        if not detail.get("passes_four_fifths")
    ]
    return {
        "four_fifths_rule": FOUR_FIFTHS,
        "attributes_audited": sorted(audit),
        "passes": not breaches,
        "breaches": breaches,
        "equalized_odds_is_stable": False,
        "detail": audit,
    }
