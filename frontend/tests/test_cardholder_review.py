"""The cardholder's request for a human review, with the API faked out."""
import pytest
from streamlit.testing.v1 import AppTest

import api

TIMEOUT = 30
ID = "id-1"
SENT_AT = "2026-11-07T10:00:00+00:00"


def _account() -> dict:
    return {
        "needs_attention": False, "reasons": [], "history": [],
        "credit_limit": 1000.0, "current_balance": 100.0, "last_payment": 100.0,
        "months_in_arrears": 0,
    }


@pytest.fixture
def faked_api(monkeypatch):
    """Returns the requests sent; `state["open"]` is what the API reports as open."""
    sent, state = [], {"open": None}

    def request_review(customer_id, token, reason):
        sent.append((customer_id, reason))
        state["open"] = {"requested_at": SENT_AT, "reason": reason}
        return state["open"]

    monkeypatch.setattr(api, "cardholder", lambda customer_id, token: _account())
    monkeypatch.setattr(
        api, "chat",
        lambda *args: {"reply": "Hello", "note": "", "disclaimer": "Careful."},
    )
    monkeypatch.setattr(api, "open_review_request", lambda customer_id, token: state["open"])
    monkeypatch.setattr(api, "request_review", request_review)
    return sent, state


def _app() -> None:
    from views import cardholder

    cardholder.render("token", "id-1")


def _start() -> AppTest:
    return AppTest.from_function(_app, default_timeout=TIMEOUT).run()


def _sent_notice(app: AppTest) -> bool:
    return any("Your request has been sent" in box.value for box in app.success)


def test_a_low_risk_customer_can_ask_for_a_review_with_a_comment(faked_api):
    sent, _ = faked_api
    app = _start()

    assert not app.exception
    app.text_area(key=f"review_comment_{ID}").set_value("Please check it.").run()
    app.button(key=f"review_send_{ID}").click().run()

    assert sent == [(ID, "Please check it.")]
    assert _sent_notice(app)


def test_a_blank_comment_is_not_sent(faked_api):
    sent, _ = faked_api
    app = _start()

    app.button(key=f"review_send_{ID}").click().run()

    assert sent == []
    assert "leave a comment" in app.error[0].value


def test_an_open_request_replaces_the_form(faked_api):
    _, state = faked_api
    state["open"] = {"requested_at": SENT_AT, "reason": "Already asked."}
    app = _start()

    assert len(app.text_area) == 0
    assert _sent_notice(app)
