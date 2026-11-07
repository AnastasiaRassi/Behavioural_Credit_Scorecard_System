"""Shapes for the officer worklist and the cardholder's own view."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QueueReason(BaseModel):
    """A driver of a score, as an officer sees it: with the effect size."""

    feature: str
    effect: float = Field(description="Log-odds; positive raises risk")
    description: str


class QueueRow(BaseModel):
    customer_id: str = Field(description="The customer's UUID")
    account_number: str = Field(description="The dataset's account number, as typed")
    full_name: str | None = None
    probability: float
    risk_band: str
    flagged: bool
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    reasons: list[QueueReason]


class MonthEntry(BaseModel):
    """One observed month. Positional, not dated: month 1 is the most recent."""

    month: int
    label: str
    payment_status: int
    payment_status_reading: str
    bill_amount: float
    amount_paid: float
    utilisation: float | None = Field(
        default=None, description="None when the credit limit is zero"
    )


class Segment(BaseModel):
    label: int
    name: str


class Highlights(BaseModel):
    """Derived from the history, never stored alongside it."""

    utilisation_now: float | None = None
    utilisation_mean: float | None = None
    trajectory: str | None = Field(
        default=None, description="improving, steady or deteriorating"
    )
    months_since_full_payment: int | None = Field(
        default=None, description="None when the bill was not cleared in six months"
    )
    months_with_no_payment: int
    worst_payment_status: int | None = None


class CustomerDetail(BaseModel):
    """The dedicated customer page: a queue row plus history, segment, highlights."""

    customer_id: str = Field(description="The customer's UUID")
    account_number: str
    full_name: str | None = None
    probability: float
    risk_band: str
    flagged: bool
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    segment: Segment | None = Field(
        default=None, description="None when segmentation has not been run"
    )
    history: list[MonthEntry]
    highlights: Highlights
    reasons: list["QueueReason"]
    latest_prediction: dict | None = Field(
        default=None,
        description="Most recent score an officer ran, or None if never run",
    )


class QueueSummary(BaseModel):
    in_queue: int
    flagged: int
    not_flagged: int
    bands: dict[str, int]


class QueuePage(BaseModel):
    """One page of the worklist, with the totals the summary tiles need.

    `summary` describes the filtered set, `summary_all` the whole queue, so a
    screen can show "41 of 4,500" without a second request.
    """

    total: int
    offset: int
    returned: int
    threshold: float
    summary: QueueSummary
    summary_all: QueueSummary
    rows: list[QueueRow]


class TriageFilter(BaseModel):
    """Which customers to pull into view. Unset fields do not filter.

    The only shape Loana's triage output is trusted in: anything the LLM writes
    outside these fields is discarded.
    """

    model_config = ConfigDict(extra="forbid")

    min_months_in_arrears: int | None = Field(default=None, ge=0, le=6)
    trajectory: Literal["improving", "steady", "deteriorating"] | None = None
    min_utilisation: float | None = Field(default=None, ge=0, le=2, description="0.9 is 90%")
    min_months_with_no_payment: int | None = Field(default=None, ge=0, le=6)
    min_credit_limit: float | None = Field(default=None, ge=0)
    max_credit_limit: float | None = Field(default=None, ge=0)
    not_scored: bool | None = Field(default=None, description="Only customers nobody has predicted yet")
    sort_by: Literal[
        "months_in_arrears", "utilisation_now", "current_balance", "months_with_no_payment"
    ] = "months_in_arrears"
    limit: int = Field(default=10, ge=1, le=25)


class CohortRow(BaseModel):
    """A customer in a triage cohort. No score unless an officer has run one."""

    customer_id: str
    account_number: str
    full_name: str | None = None
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    utilisation_now: float | None = None
    trajectory: str | None = None
    months_with_no_payment: int
    last_risk_band: str | None = Field(default=None, description="From the latest prediction, if any")
    last_scored_at: str | None = None


class CohortPage(BaseModel):
    description: str = Field(description="The filter in plain words")
    total: int = Field(description="Matching customers before the limit")
    rows: list[CohortRow]


class TriageRequest(BaseModel):
    """The triage conversation so far, ending with the officer's request."""

    messages: list["ChatMessage"] = Field(min_length=1, max_length=40)


class TriageReply(BaseModel):
    reply: str
    source: str = Field(description="model, or template when no usable filter came back")
    note: str = ""
    filter: TriageFilter | None = None
    cohort: CohortPage | None = None


class BulkPredictRequest(BaseModel):
    customer_ids: list[str] = Field(min_length=1, max_length=10)
    scored_by: str = Field(description="Email of the officer running the predictions")


class CardholderReason(BaseModel):
    """A driver of a score, as the cardholder sees it: no effect size."""

    feature: str
    description: str
    raises_risk: bool


class CardholderMonth(BaseModel):
    """One billed month for the cardholder's chart. No probability, no effect sizes."""

    month: int
    label: str
    payment_status: int
    payment_status_reading: str
    bill_amount: float
    amount_paid: float


class CardholderCoaching(BaseModel):
    text: str
    disclaimer: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    """The conversation so far, oldest first. Empty asks for the opening interpretation."""

    messages: list[ChatMessage] = Field(default_factory=list, max_length=40)


class ChatReply(BaseModel):
    reply: str
    source: str = Field(description="model, or template when the LLM was unavailable or its reply failed a check")
    note: str = Field(default="", description="Shown under a template reply; empty for model replies")
    disclaimer: str


class CardholderView(BaseModel):
    """Deliberately narrower than QueueRow.

    No probability and no threshold: the number is not sent, not merely hidden.
    """

    customer_id: str
    account_number: str
    risk_band: str
    needs_attention: bool
    credit_limit: float
    current_balance: float
    last_payment: float
    months_in_arrears: int
    reasons: list[CardholderReason]
    coaching: CardholderCoaching
    history: list[CardholderMonth]


class RoleOut(BaseModel):
    role: str
    label: str
    description: str
    landing_endpoint: str
    users: int
