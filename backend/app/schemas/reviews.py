"""Shapes for cardholder messages to the officer and the shared thread."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ReviewRequestCreate(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator("reason")
    @classmethod
    def reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason cannot be blank")
        return value


class ReviewRequestOut(BaseModel):
    request_id: str
    customer_id: str
    account_number: str
    full_name: str | None = None
    requested_at: str
    reason: str
    status: str


class ThreadEntry(BaseModel):
    """One card in the shared thread: a cardholder message or an officer decision."""

    id: str
    kind: Literal["message", "decision"]
    at: str
    author: str
    body: str
    # Decisions only.
    action: str | None = None
    # Decisions only: the risk level the officer set, if they set one.
    band: str | None = None
    # Messages only: pending, in_review, resolved or rejected.
    status: str | None = None


class ReviewAction(BaseModel):
    """Who is accepting or declining a customer's open requests."""

    decided_by: str = Field(max_length=255, description="Email of the officer")
