"""Smoke checks for the HTTP surface.

Every test runs against a session bound to a transaction that is rolled back, so
the suite leaves no rows behind even though the endpoints write. The scorer and
artifacts are real: these tests exercise the same code path a request takes.

Skipped when no database is reachable.
"""
import io
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.dependencies import get_session
from app.db.database import engine
from app.db.models import ModelVersion, User
from app.main import app
from app.ml.scorer import example_customer

OFFICER_EMAIL = "test-officer@example.test"


@pytest.fixture
def client():
    try:
        connection = engine().connect()
    except OperationalError as exception:
        pytest.skip(f"no database reachable: {exception}")

    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)

    # A live model version and an officer must exist for scoring to work. Created
    # inside the transaction so seeded state is not required.
    session.query(ModelVersion).filter_by(is_live=True).update({"is_live": False})
    session.add(ModelVersion(
        version="api_test_version",
        trained_on=datetime.now(timezone.utc),
        is_live=True,
    ))
    session.add(User(email=OFFICER_EMAIL, display_name="Test Officer", role="officer"))
    session.flush()

    # The endpoints must not commit or close the outer transaction.
    app.dependency_overrides[get_session] = lambda: session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        session.close()
        transaction.rollback()
        connection.close()


def _score(client, **overrides):
    payload = {"customer": example_customer(), "scored_by": OFFICER_EMAIL}
    payload.update(overrides)
    return client.post("/predictions", json=payload)


def test_health_reports_the_served_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] == 46


def test_scoring_returns_a_band_and_reasons(client):
    response = _score(client, cardholder_reference="Cardholder 7001")
    assert response.status_code == 201

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert body["flagged"] == (body["probability"] >= body["threshold"])
    assert body["reasons"], "a score with no reasons is not explainable"
    assert all(reason["description"] for reason in body["reasons"])


def test_reasons_never_name_a_protected_attribute(client):
    body = _score(client).json()
    named = " ".join(reason["feature"] for reason in body["reasons"])
    for protected in ("SEX", "AGE", "EDUCATION", "MARRIAGE"):
        assert protected not in named


def test_a_stored_prediction_keeps_the_reasons_it_showed(client):
    created = _score(client).json()
    stored = client.get(f"/predictions/{created['prediction_id']}").json()

    assert stored["probability"] == created["probability"]
    shown = [reason["feature"] for reason in created["reasons"]]
    kept = [reason["feature"] for reason in stored["explanation"]["reasons"]]
    assert kept == shown


def test_an_invalid_customer_is_rejected(client):
    customer = example_customer()
    customer["PAY_1"] = 99  # outside the published status range
    response = client.post(
        "/predictions", json={"customer": customer, "scored_by": OFFICER_EMAIL}
    )
    assert response.status_code == 422


def test_an_unknown_officer_is_rejected(client):
    response = _score(client, scored_by="nobody@example.test")
    assert response.status_code == 404


def test_a_decision_records_who_acted_and_why(client):
    prediction_id = _score(client).json()["prediction_id"]
    response = client.post(
        f"/predictions/{prediction_id}/decisions",
        json={
            "action": "contact_customer",
            "justification": "Two months in arrears; calling the customer.",
            "decided_by": OFFICER_EMAIL,
        },
    )
    assert response.status_code == 201
    assert response.json()["action"] == "contact_customer"


def test_a_decision_needs_a_real_justification(client):
    prediction_id = _score(client).json()["prediction_id"]
    response = client.post(
        f"/predictions/{prediction_id}/decisions",
        json={"action": "monitor", "justification": "   ", "decided_by": OFFICER_EMAIL},
    )
    assert response.status_code == 422


def test_loan_vocabulary_is_not_accepted(client):
    """The model scores existing cardholders, so there is nothing to approve."""
    prediction_id = _score(client).json()["prediction_id"]
    response = client.post(
        f"/predictions/{prediction_id}/decisions",
        json={"action": "approve", "justification": "x", "decided_by": OFFICER_EMAIL},
    )
    assert response.status_code == 422


def test_decisions_accumulate_on_one_prediction(client):
    prediction_id = _score(client).json()["prediction_id"]
    for action in ("monitor", "contact_customer"):
        client.post(
            f"/predictions/{prediction_id}/decisions",
            json={
                "action": action,
                "justification": f"escalating: {action}",
                "decided_by": OFFICER_EMAIL,
            },
        )

    listed = client.get(f"/predictions/{prediction_id}/decisions").json()
    assert [row["action"] for row in listed] == ["monitor", "contact_customer"]


def test_a_missing_prediction_is_a_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/predictions/{missing}").status_code == 404


def test_the_report_matches_the_saved_artifacts(client):
    body = client.get("/report").json()
    assert body["n_features"] == 46
    assert body["test_rows"] == 4500
    assert 0.77 <= body["test_metrics"]["roc_auc"] <= 0.79


def test_fairness_reports_a_pass_and_flags_unstable_columns(client):
    body = client.get("/fairness").json()
    assert body["passes"] is True
    assert body["breaches"] == []
    assert body["equalized_odds_is_stable"] is False
    assert set(body["attributes_audited"]) == {"SEX", "AGE", "EDUCATION", "MARRIAGE"}


def test_drift_declines_to_report_on_too_few_rows(client):
    body = client.get("/monitoring/drift").json()
    assert body["sufficient"] is False
    assert "too few" in body["reason"]


def test_performance_declines_to_report_without_labels(client):
    body = client.get("/monitoring/performance").json()
    assert body["sufficient"] is False
    assert body["labelled_predictions"] == 0


def test_a_csv_batch_scores_every_row(client):
    customer = example_customer()
    header = ",".join(customer)
    line = ",".join(str(value) for value in customer.values())
    csv = f"{header}\n{line}\n{line}\n"

    response = client.post(
        "/predictions/batch",
        files={"file": ("batch.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scored"] == 2
    assert body["rejected"] == 0


def test_a_csv_missing_columns_is_rejected(client):
    response = client.post(
        "/predictions/batch",
        files={"file": ("bad.csv", io.BytesIO(b"LIMIT_BAL\n1000\n"), "text/csv")},
    )
    assert response.status_code == 422
    assert "Missing required columns" in response.json()["detail"]
