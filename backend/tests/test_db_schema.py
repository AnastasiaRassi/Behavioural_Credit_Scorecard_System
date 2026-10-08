"""Smoke checks for the output schema: the guarantees the database itself enforces.

Each case runs inside a savepoint that is rolled back, so the tests leave no rows
behind and can run against a database holding real demo data.

Skipped when no database is reachable, so the rest of the suite still runs on a
machine with no Postgres.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from app.db.database import _session_factory, engine
from app.db.models import Decision, ModelVersion, Prediction, User


@pytest.fixture(scope="module")
def session():
    try:
        with engine().connect() as connection:
            connection.execute(text("select 1"))
    except OperationalError as exception:
        pytest.skip(f"no database reachable: {exception}")

    active = _session_factory()()
    try:
        yield active
    finally:
        # Nothing these tests wrote is kept.
        active.rollback()
        active.close()


@pytest.fixture
def seeded(session):
    """A valid officer, subject, live model version and prediction.

    Created in a savepoint the test rolls back, so each test starts from the same
    state regardless of order.
    """
    savepoint = session.begin_nested()

    officer = User(display_name="Demo Officer", role="officer")
    subject = User(display_name="Cardholder 1001", role="cardholder")
    # Not live: a seeded live model may already exist, and only one is allowed.
    version = ModelVersion(
        version="test_version",
        trained_on=datetime(2026, 10, 3, tzinfo=timezone.utc),
        is_live=False,
    )
    session.add_all([officer, subject, version])
    session.flush()

    prediction = Prediction(
        subject_user_id=subject.id,
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


def test_a_valid_chain_inserts(seeded):
    assert seeded["prediction"].id is not None
    assert seeded["prediction"].subject_user_id == seeded["subject"].id


def test_jsonb_survives_a_round_trip(session, seeded):
    session.refresh(seeded["prediction"])
    stored = seeded["prediction"].explanation
    assert stored["contributions"][0]["feature"] == "pay_max"
    assert stored["base_value"] == -1.2


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


def test_roles_are_restricted(session, seeded):
    message = _rejected_by(session, User(display_name="X", role="auditor"))
    assert "role_is_known" in message


def test_a_probability_must_be_a_rate(session, seeded):
    message = _rejected_by(session, Prediction(
        subject_user_id=seeded["subject"].id,
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
        version="test_version_2",
        trained_on=datetime.now(timezone.utc),
        is_live=True,
    ))
    assert "index_one_live_model_version" in message


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
