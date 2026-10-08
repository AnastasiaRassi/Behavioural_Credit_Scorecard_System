"""Request and response shapes for scoring.

Field names keep the dataset's original casing (`LIMIT_BAL`, `PAY_1`), because
`feature_engineering.engineer()` reads those exact column names. Renaming them
here would mean mapping back before scoring, which is the kind of second
translation that eventually disagrees with training.
"""
from pydantic import BaseModel, Field

# Repayment status codes as published with the dataset: -2 no balance,
# -1 paid in full, 0 revolving credit, 1..9 months in arrears.
PAY_STATUS_MINIMUM = -2
PAY_STATUS_MAXIMUM = 9


class CustomerInput(BaseModel):
    """Six months of a cardholder's billing history, as the dataset records it.

    The four protected attributes are accepted because the fairness audit needs
    them, and are dropped inside `CreditScorer.prepare()` before anything reaches
    the model. A caller cannot cause them to be used in a decision.
    """

    LIMIT_BAL: float = Field(gt=0, description="Credit limit, NT dollars")

    SEX: int = Field(ge=1, le=2, description="Protected; audited, never scored")
    AGE: int = Field(ge=18, le=120, description="Protected; audited, never scored")
    EDUCATION: int = Field(ge=0, le=6, description="Protected; audited, never scored")
    MARRIAGE: int = Field(ge=0, le=3, description="Protected; audited, never scored")

    PAY_1: int = Field(ge=PAY_STATUS_MINIMUM, le=PAY_STATUS_MAXIMUM)
    PAY_2: int = Field(ge=PAY_STATUS_MINIMUM, le=PAY_STATUS_MAXIMUM)
    PAY_3: int = Field(ge=PAY_STATUS_MINIMUM, le=PAY_STATUS_MAXIMUM)
    PAY_4: int = Field(ge=PAY_STATUS_MINIMUM, le=PAY_STATUS_MAXIMUM)
    PAY_5: int = Field(ge=PAY_STATUS_MINIMUM, le=PAY_STATUS_MAXIMUM)
    PAY_6: int = Field(ge=PAY_STATUS_MINIMUM, le=PAY_STATUS_MAXIMUM)

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class ScoreRequest(BaseModel):
    """A customer to score, and who they are, if the caller knows.

    `cardholder_reference` names an existing cardholder so repeated scores join
    up over time. Omitted, a fresh cardholder row is created, which is what an
    officer keying in a walk-in wants.
    """

    customer: CustomerInput
    cardholder_reference: str | None = Field(default=None, max_length=255)
    scored_by: str | None = Field(
        default=None, max_length=255, description="Email of the officer scoring"
    )


class ReasonOut(BaseModel):
    """One driver of a score, in plain language."""

    feature: str
    value: float
    effect: float = Field(description="Log-odds; positive raises risk")
    description: str
    raises_risk: bool


class ScoreResponse(BaseModel):
    prediction_id: str
    probability: float
    risk_band: str
    threshold: float
    flagged: bool
    band_default_rate: float
    model_version: str
    reasons: list[ReasonOut]


class BatchRow(BaseModel):
    row: int
    probability: float | None = None
    risk_band: str | None = None
    flagged: bool | None = None
    error: str | None = None


class BatchScoreResponse(BaseModel):
    scored: int
    rejected: int
    rows: list[BatchRow]
