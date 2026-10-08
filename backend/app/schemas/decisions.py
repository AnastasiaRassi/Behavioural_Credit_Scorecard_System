"""Request and response shapes for officer decisions."""
from pydantic import BaseModel, Field, field_validator

from app.db.models import DECISION_ACTIONS


class DecisionCreate(BaseModel):
    """An officer's intervention on a scored account.

    The justification is mandatory and must say something. An audit trail with
    blank entries is harder to defend than one that is always complete, so the
    database rejects empty text too; this check only turns it into a 422 rather
    than a 500.
    """

    action: str
    justification: str = Field(min_length=1, max_length=2000)
    decided_by: str = Field(max_length=255, description="Email of the deciding officer")

    @field_validator("action")
    @classmethod
    def action_is_known(cls, value: str) -> str:
        if value not in DECISION_ACTIONS:
            raise ValueError(
                f"action must be one of {', '.join(DECISION_ACTIONS)}, got {value!r}"
            )
        return value

    @field_validator("justification")
    @classmethod
    def justification_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("justification cannot be blank")
        return value


class DecisionOut(BaseModel):
    id: str
    prediction_id: str
    decided_at: str
    action: str
    justification: str
    overrode_model: bool = Field(
        description="True when the officer acted against what the score suggested"
    )
