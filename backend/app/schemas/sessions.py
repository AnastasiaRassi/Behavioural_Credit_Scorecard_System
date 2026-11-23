"""Shapes for signing in."""
from typing import Literal

from pydantic import BaseModel, Field


class Credentials(BaseModel):
    """What a sign-in attempt submits. Never logged and never stored."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class Account(BaseModel):
    """Who the caller is. The role is read from the account, never submitted."""

    email: str
    display_name: str
    role: str
    customer_id: str | None = Field(
        default=None,
        description="The customer's UUID for a cardholder; None for staff",
    )
    # The caller's own record, for their profile page. Cardholders only.
    account_number: str | None = None
    age_years: int | None = None
    sex: int | None = None
    education: int | None = None
    marriage: int | None = None


class ProfileEdit(BaseModel):
    """What a cardholder may change about themselves. Bounds mirror the DB checks."""

    # Name, age and sex are legal identity data and deliberately not editable.
    # Codes are the codebook's; anything else is rejected, not coerced.
    education: Literal[1, 2, 3, 4]
    marriage: Literal[1, 2, 3]


class SignedIn(Account):
    token: str
    expires_at: int = Field(description="Epoch seconds; the token is rejected after")
