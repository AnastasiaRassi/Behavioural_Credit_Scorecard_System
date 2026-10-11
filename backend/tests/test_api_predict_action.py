"""The officer's "predict default" action, and what each role then sees.

The point of these checks is that the officer and the cardholder end up looking
at the same prediction, while still being sent different payloads.
"""
import pytest

from app.ml import worklist
from tests.test_api import OFFICER_EMAIL, client  # noqa: F401  (fixture import)


@pytest.fixture(scope="module")
def customer_id() -> str:
    return worklist.all_rows()[0]["customer_id"]


def _predict(client, customer_id: str, officer: str = OFFICER_EMAIL):
    return client.post(
        f"/officer/customers/{customer_id}/predict", params={"scored_by": officer}
    )


def test_an_officer_can_run_a_prediction(client, customer_id):
    response = _predict(client, customer_id)
    assert response.status_code == 201

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert body["flagged"] == (body["probability"] >= body["threshold"])
    assert body["model_version"]


def test_the_prediction_comes_with_its_explanation(client, customer_id):
    body = _predict(client, customer_id).json()

    assert body["reasons"], "a prediction with no reasons is not explainable"
    assert all(reason["description"] for reason in body["reasons"])
    assert all("raises_risk" in reason for reason in body["reasons"])


def test_a_fresh_prediction_agrees_with_the_queue(client, customer_id):
    """Scoring blind reaches the same number the overnight queue recorded."""
    queued = client.get(f"/officer/customers/{customer_id}").json()
    fresh = _predict(client, customer_id).json()

    assert fresh["probability"] == pytest.approx(queued["probability"], abs=1e-4)
    assert fresh["risk_band"] == queued["risk_band"]


def test_the_prediction_is_stored_and_readable(client, customer_id):
    created = _predict(client, customer_id).json()
    stored = client.get(f"/predictions/{created['prediction_id']}").json()

    assert stored["probability"] == pytest.approx(created["probability"])
    assert stored["explanation"]["reasons"]


def test_the_detail_page_reports_the_latest_prediction(client, customer_id):
    assert client.get(f"/officer/customers/{customer_id}").json()[
        "latest_prediction"
    ] is None

    created = _predict(client, customer_id).json()
    latest = client.get(f"/officer/customers/{customer_id}").json()["latest_prediction"]

    assert latest is not None
    assert latest["id"] == created["prediction_id"]


def test_running_it_twice_keeps_both_scores(client, customer_id):
    first = _predict(client, customer_id).json()
    second = _predict(client, customer_id).json()

    assert first["prediction_id"] != second["prediction_id"]
    latest = client.get(f"/officer/customers/{customer_id}").json()["latest_prediction"]
    assert latest["id"] == second["prediction_id"]


def test_the_cardholder_sees_the_officer_prediction(client, customer_id):
    _predict(client, customer_id)
    cardholder = client.get(f"/cardholders/{customer_id}").json()
    officer_view = client.get(f"/officer/customers/{customer_id}").json()

    assert cardholder["risk_band"] == officer_view["latest_prediction"]["risk_band"]
    assert cardholder["needs_attention"] == officer_view["latest_prediction"]["flagged"]
    assert cardholder["reasons"]


def test_the_cardholder_still_is_not_sent_the_number(client, customer_id):
    _predict(client, customer_id)
    body = client.get(f"/cardholders/{customer_id}").json()

    assert "probability" not in body
    assert "threshold" not in body
    assert all("effect" not in reason for reason in body["reasons"])


def test_a_decision_attaches_to_the_prediction_that_was_run(client, customer_id):
    created = _predict(client, customer_id).json()
    decision = client.post(
        f"/officer/customers/{customer_id}/decisions",
        json={
            "action": "contact_customer",
            "justification": "Score run today; calling the customer.",
            "decided_by": OFFICER_EMAIL,
        },
    ).json()

    assert decision["prediction_id"] == created["prediction_id"]


def test_predicting_needs_a_known_officer(client, customer_id):
    assert _predict(client, customer_id, "nobody@example.test").status_code == 404


def test_predicting_an_unknown_customer_is_a_404(client):
    assert _predict(client, "C-nobody").status_code == 404


def test_the_stored_input_withholds_protected_attributes(client, customer_id):
    created = _predict(client, customer_id).json()
    stored = client.get(f"/predictions/{created['prediction_id']}").json()

    # The read payload does not expose raw_input, so check the record directly.
    from app.db.queue_predictions import raw_input_from_history

    raw = raw_input_from_history(worklist.find(customer_id))
    for protected in ("SEX", "AGE", "EDUCATION", "MARRIAGE"):
        assert protected not in raw
    assert raw["_protected_attributes_withheld"] is True
    assert stored["risk_band"] in {"Low", "Medium", "High"}
