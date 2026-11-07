"""The cardholder's message thread with the loan officer, with the API faked out."""
import pytest
from streamlit.testing.v1 import AppTest

import api

TIMEOUT = 30
ID = "id-1"
SENT_AT = "2026-11-07T10:00:00+00:00"


def _account() -> dict:
    return {
        "needs_attention": False, "risk_band": "Low", "reasons": [], "history": [],
        "credit_limit": 1000.0, "current_balance": 100.0, "last_payment": 100.0,
        "months_in_arrears": 0,
    }


def _message(body: str, status: str = "pending") -> dict:
    return {
        "id": body, "kind": "message", "at": SENT_AT, "author": "Maya",
        "body": body, "action": None, "status": status,
    }


def _decision(body: str) -> dict:
    return {
        "id": body, "kind": "decision", "at": SENT_AT, "author": "Officer",
        "body": body, "action": "monitor", "status": None,
    }


@pytest.fixture
def faked_api(monkeypatch):
    """Returns the messages sent and the thread the API reports."""
    sent, entries = [], []

    def request_review(customer_id, token, reason):
        sent.append((customer_id, reason))
        entries.append(_message(reason))
        return {}

    monkeypatch.setattr(api, "cardholder", lambda customer_id, token: _account())
    monkeypatch.setattr(
        api, "chat",
        lambda *args: {"reply": "Hello", "note": "", "disclaimer": "Careful."},
    )
    monkeypatch.setattr(api, "cardholder_thread", lambda customer_id, token: list(entries))
    monkeypatch.setattr(api, "request_review", request_review)
    return sent, entries


def _app() -> None:
    from views import cardholder

    cardholder.render("token", "id-1")


def _start() -> AppTest:
    return AppTest.from_function(_app, default_timeout=TIMEOUT).run()


def _captions(app: AppTest) -> str:
    return " ".join(caption.value for caption in app.caption)


def _markdown(app: AppTest) -> str:
    return " ".join(element.value for element in app.markdown)


def test_a_message_is_sent_shown_in_the_thread_and_the_box_cleared(faked_api):
    sent, _ = faked_api
    app = _start()

    assert not app.exception
    app.text_area(key=f"review_comment_{ID}").set_value("Please check it.").run()
    app.button(key=f"review_send_{ID}").click().run()

    assert sent == [(ID, "Please check it.")]
    assert "Please check it." in _markdown(app)
    assert "reply here" in _captions(app)
    assert app.text_area(key=f"review_comment_{ID}").value == ""


def test_a_blank_message_is_not_sent(faked_api):
    sent, _ = faked_api
    app = _start()

    app.button(key=f"review_send_{ID}").click().run()

    assert sent == []
    assert "write a message" in app.error[0].value


def test_a_decision_closes_the_conversation_but_the_box_stays(faked_api):
    _, entries = faked_api
    entries.extend([_message("Already asked.", "resolved"), _decision("Monitoring it.")])
    app = _start()

    text = _markdown(app)
    assert "Already asked." in text and "Monitoring it." in text and "Decision" in text
    assert "conversation is closed" in _captions(app)
    assert app.text_area(key=f"review_comment_{ID}") is not None


def test_message_bodies_are_escaped(faked_api):
    _, entries = faked_api
    entries.append(_message("<script>alert(1)</script>"))
    app = _start()

    assert "<script>" not in _markdown(app)
