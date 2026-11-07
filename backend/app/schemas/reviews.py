"""Request and response shapes for human-review requests."""
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
