"""Smoke checks for the schema: the guarantees the database itself enforces.

Each case runs inside a savepoint that is rolled back, so the tests leave no rows
behind and can run against a database holding real demo data.

Skipped when no database is reachable, so the rest of the suite still runs on a
machine with no Postgres.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.passwords import hash_password
from app.db.database import _session_factory
from app.db.models import (
    CreditOutcome,
    Customer,
    Decision,
    ModelVersion,
    MonthlyStatement,
    Prediction,
    UserAccount,
)
from tests.conftest import connect_or_skip

# Prefixed so a test row can never be mistaken for an ingested source id.
TEST_CUSTOMER_ID = "TEST-1001"


@pytest.fixture(scope="module")
def session():
    connect_or_skip().close()

    active = _session_factory()()
    try:
        yield active
    finally:
        # Nothing these tests wrote is kept.
        active.rollback()
        active.close()


def _account(role: str, display_name: str) -> UserAccount:
    return UserAccount(
        email=f"{uuid.uuid4().hex}@example.test",
        display_name=display_name,
        password_hash=hash_password("not-a-real-password"),
        role=role,
    )


@pytest.fixture
def seeded(session):
    """An officer, a customer with a full history, a model version and a prediction.

    Created in a savepoint the test rolls back, so each test starts from the same
    state regardless of order.
    """
    savepoint = session.begin_nested()

    officer = _account("officer", "Demo Officer")
    subject = Customer(
        customer_id=TEST_CUSTOMER_ID,
        credit_limit=20000,
        sex=2,
        age_years=24,
        education=2,
        marriage=1,
    )
    # Not live: a seeded live model may already exist, and only one is allowed.
    version = ModelVersion(
        version=f"test_version_{uuid.uuid4().hex[:8]}",
        trained_on=datetime(2026, 10, 3, tzinfo=timezone.utc),
        is_live=False,
    )
    session.add_all([officer, subject, version])
    session.flush()

    for month in range(1, 7):
        session.add(
            MonthlyStatement(
                customer_id=subject.customer_id,
                month_offset=month,
                repayment_status=2 if month == 1 else 0,
                bill_amount=3913 - month,
                payment_amount=month * 100,
            )
        )
    session.add(
        CreditOutcome(
            customer_id=subject.customer_id, defaulted=True, observed_period="2005-10"
        )
    )

    prediction = Prediction(
        subject_customer_id=subject.customer_id,
        model_version_id=version.id,
        probability=0.61,
        risk_band="High",
        flagged=True,
        threshold=0.2747,
        raw_input={"LIMIT_BAL": 20000, "PAY_1": 2},
        explanation={
            "base_value": -1.2,
            "contributions": [{"feature": "pay_max", "value": 2.0, "effect": 0.84}],
        },
    )
    session.add(prediction)
    session.flush()

    yield {"officer": officer, "subject": subject, "version": version,
           "prediction": prediction}

    savepoint.rollback()


def _rejected_by(session, instance) -> str:
    """Add an instance expected to violate a constraint; return the message."""
    savepoint = session.begin_nested()
    try:
        session.add(instance)
        session.flush()
    except IntegrityError as exception:
        return str(exception.orig)
    else:
        pytest.fail(f"{type(instance).__name__} was accepted but should not be")
    finally:
        savepoint.rollback()


def _customer(**overrides) -> Customer:
    values = {
        "customer_id": f"TEST-{uuid.uuid4().hex[:8]}",
        "credit_limit": 50000,
        "sex": 1,
        "age_years": 40,
        "education": 1,
        "marriage": 1,
    }
    return Customer(**{**values, **overrides})


def _statement(customer_id: str, **overrides) -> MonthlyStatement:
    values = {
        "customer_id": customer_id,
        "month_offset": 1,
        "repayment_status": 0,
        "bill_amount": 1000,
        "payment_amount": 100,
    }
    return MonthlyStatement(**{**values, **overrides})


# ---- the chain holds together ------------------------------------------------


def test_a_valid_chain_inserts(seeded):
    assert seeded["prediction"].id is not None
    assert seeded["prediction"].subject_customer_id == seeded["subject"].customer_id


def test_a_customer_keeps_its_six_months_in_order(session, seeded):
    session.refresh(seeded["subject"])
    months = [row.month_offset for row in seeded["subject"].statements]
    assert months == [1, 2, 3, 4, 5, 6]


def test_jsonb_survives_a_round_trip(session, seeded):
    session.refresh(seeded["prediction"])
    stored = seeded["prediction"].explanation
    assert stored["contributions"][0]["feature"] == "pay_max"
    assert stored["base_value"] == -1.2


# ---- the separation the schema exists to enforce -----------------------------


def test_a_customer_needs_no_account(seeded):
    assert seeded["subject"].user_account_id is None
    assert seeded["subject"].is_registered is False


def test_one_account_cannot_claim_two_customers(session, seeded):
    cardholder = _account("cardholder", "Registered Cardholder")
    session.add(cardholder)
    session.flush()

    seeded["subject"].user_account_id = cardholder.id
    session.flush()

    message = _rejected_by(session, _customer(user_account_id=cardholder.id))
    assert "user_account_id" in message


def test_a_prediction_cannot_point_at_an_unknown_customer(session, seeded):
    message = _rejected_by(session, Prediction(
        subject_customer_id="TEST-never-ingested",
        model_version_id=seeded["version"].id,
        probability=0.5,
        risk_band="Medium",
        flagged=False,
        threshold=0.2747,
        raw_input={},
        explanation={},
    ))
    assert "subject_customer_id" in message or "foreign key" in message.lower()


# ---- customer attribute codes ------------------------------------------------


def test_a_credit_limit_must_be_positive(session, seeded):
    message = _rejected_by(session, _customer(credit_limit=0))
    assert "credit_limit_is_positive" in message


def test_sex_codes_are_restricted(session, seeded):
    message = _rejected_by(session, _customer(sex=3))
    assert "sex_is_a_codebook_value" in message


def test_an_age_must_be_plausible(session, seeded):
    message = _rejected_by(session, _customer(age_years=17))
    assert "age_years_is_plausible" in message


def test_undocumented_education_codes_are_rejected(session, seeded):
    # 0, 5 and 6 are folded to 4 at ingest; reaching the database means the
    # folding was skipped.
    message = _rejected_by(session, _customer(education=5))
    assert "education_is_a_codebook_value" in message


def test_undocumented_marriage_codes_are_rejected(session, seeded):
    message = _rejected_by(session, _customer(marriage=0))
    assert "marriage_is_a_codebook_value" in message


# ---- monthly statements ------------------------------------------------------


def test_a_month_offset_must_be_within_the_window(session, seeded):
    message = _rejected_by(
        session, _statement(seeded["subject"].customer_id, month_offset=7)
    )
    assert "month_offset_is_within_the_window" in message


def test_a_month_cannot_be_recorded_twice(session, seeded):
    message = _rejected_by(
        session, _statement(seeded["subject"].customer_id, month_offset=1)
    )
    assert "monthly_statement_pkey" in message


def test_repayment_status_codes_are_restricted(session, seeded):
    message = _rejected_by(
        session, _statement(seeded["subject"].customer_id, repayment_status=99)
    )
    assert "repayment_status_is_a_codebook_value" in message


def test_a_payment_cannot_be_negative(session, seeded):
    message = _rejected_by(
        session,
        _statement(seeded["subject"].customer_id, month_offset=1, payment_amount=-1),
    )
    assert "payment_amount_is_not_negative" in message


def test_a_negative_bill_is_accepted(session, seeded):
    """An account in credit has a negative bill, so it must not be constrained."""
    savepoint = session.begin_nested()
    customer = _customer()
    session.add(customer)
    session.flush()
    session.add(_statement(customer.customer_id, bill_amount=-250))
    session.flush()
    savepoint.rollback()


# ---- accounts ----------------------------------------------------------------


def test_roles_are_restricted(session, seeded):
    message = _rejected_by(session, _account("auditor", "X"))
    assert "role_is_known" in message


def test_an_email_cannot_be_reused(session, seeded):
    message = _rejected_by(session, UserAccount(
        email=seeded["officer"].email,
        display_name="Impostor",
        password_hash=hash_password("not-a-real-password"),
        role="officer",
    ))
    assert "user_account_email_key" in message


def test_an_account_cannot_store_an_empty_hash(session, seeded):
    account = _account("officer", "No Credential")
    account.password_hash = ""
    message = _rejected_by(session, account)
    assert "password_hash_is_present" in message


def test_an_account_payload_never_carries_the_hash(seeded):
    assert "password_hash" not in seeded["officer"].to_dict()


# ---- decisions and outcomes --------------------------------------------------


def test_a_decision_needs_a_justification(session, seeded):
    message = _rejected_by(session, Decision(
        prediction_id=seeded["prediction"].id,
        decided_by_user_id=seeded["officer"].id,
        action="monitor",
        justification="   ",
    ))
    assert "justification_is_present" in message


def test_decision_actions_are_restricted(session, seeded):
    message = _rejected_by(session, Decision(
        prediction_id=seeded["prediction"].id,
        decided_by_user_id=seeded["officer"].id,
        action="approve",
        justification="not a behavioural action",
    ))
    assert "action_is_known" in message


def test_a_probability_must_be_a_rate(session, seeded):
    message = _rejected_by(session, Prediction(
        subject_customer_id=seeded["subject"].customer_id,
        model_version_id=seeded["version"].id,
        probability=1.4,
        risk_band="High",
        flagged=True,
        threshold=0.2747,
        raw_input={},
        explanation={},
    ))
    assert "probability_is_a_rate" in message


def test_only_one_model_version_can_be_live(session, seeded):
    # Demote whatever is live first, so the test holds whether or not the
    # database has been seeded.
    session.query(ModelVersion).filter_by(is_live=True).update({"is_live": False})
    session.flush()
    seeded["version"].is_live = True
    session.flush()

    message = _rejected_by(session, ModelVersion(
        version=f"test_version_{uuid.uuid4().hex[:8]}",
        trained_on=datetime.now(timezone.utc),
        is_live=True,
    ))
    assert "index_one_live_model_version" in message


def test_a_dataset_label_cannot_be_recorded_twice(session, seeded):
    message = _rejected_by(session, CreditOutcome(
        customer_id=seeded["subject"].customer_id,
        defaulted=False,
        observed_period="2005-10",
    ))
    assert "credit_outcome_pkey" in message


def test_an_outcome_cannot_be_recorded_twice(session, seeded):
    from app.db.models import ObservedOutcome

    observed = ObservedOutcome(
        prediction_id=seeded["prediction"].id,
        observed_at=datetime.now(timezone.utc),
        defaulted=True,
    )
    session.add(observed)
    session.flush()

    message = _rejected_by(session, ObservedOutcome(
        prediction_id=seeded["prediction"].id,
        observed_at=datetime.now(timezone.utc),
        defaulted=False,
    ))
    assert "observed_outcome" in message


# ---- the chat log --------------------------------------------------------------


def _chat_turn(**overrides):
    from app.db.models import ChatTurn

    values = dict(
        customer_id="TEST-chat", turn_number=1, user_message="Why?", reply="Because.",
        source="model", llm_model="gpt-4o-mini", prompt_version="v2.1-loana",
        scoring_model_version="m1", latency_ms=900,
    )
    return ChatTurn(**{**values, **overrides})


@pytest.fixture
def chat_table(session):
    """chat_turn inside a savepoint, so a database that never had it still runs these."""
    from app.db.models import ChatTurn

    savepoint = session.begin_nested()
    ChatTurn.__table__.create(session.connection(), checkfirst=True)
    yield session
    savepoint.rollback()


def test_the_guard_columns_are_nullable_booleans(chat_table):
    from sqlalchemy import Boolean

    from app.db.models import ChatTurn

    for name in ("grounding_passed", "claim_passed"):
        column = ChatTurn.__table__.c[name]
        assert isinstance(column.type, Boolean) and column.nullable

    stored = _chat_turn(grounding_passed=True)
    chat_table.add(stored)
    chat_table.flush()
    assert (stored.grounding_passed, stored.claim_passed) == (True, None)


def test_every_late_added_column_is_on_the_model():
    # create_all adds these to a pre-existing table; a typo here would add a stray column.
    from app.db.database import ADDED_COLUMNS
    from app.db.models import Base

    for table, column, _ in ADDED_COLUMNS:
        assert column in Base.metadata.tables[table].c


def test_the_live_chat_turn_table_has_the_guard_columns(session):
    from sqlalchemy import inspect

    inspector = inspect(session.connection())
    if "chat_turn" not in inspector.get_table_names():
        pytest.skip("chat_turn not created yet; run: python -m app.db.seed")
    columns = {column["name"] for column in inspector.get_columns("chat_turn")}
    assert {"grounding_passed", "claim_passed"} <= columns


def test_the_opening_fits_the_table_as_turn_zero_with_an_empty_message(chat_table):
    opening = _chat_turn(turn_number=0, user_message="")
    chat_table.add(opening)
    chat_table.flush()

    assert opening.id is not None


def test_a_template_turn_still_needs_its_reason(chat_table):
    message = _rejected_by(chat_table, _chat_turn(source="template", fallback_reason=None))

    assert "fallback_reason_iff_template" in message
