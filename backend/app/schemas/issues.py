"""Shapes for the Report an Issue page."""
from pydantic import BaseModel, Field, field_validator

from app.db.models import ISSUE_KINDS


class IssueReportCreate(BaseModel):
    kind: str
    details: str = Field(min_length=1, max_length=2000)

    @field_validator("kind")
    @classmethod
    def kind_is_known(cls, value: str) -> str:
        if value not in ISSUE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(ISSUE_KINDS)}, got {value!r}")
        return value

    @field_validator("details")
    @classmethod
    def details_are_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("details cannot be blank")
        return value


class IssueReportOut(BaseModel):
    id: str
    created_at: str
