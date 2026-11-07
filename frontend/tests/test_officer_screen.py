"""The officer screen end to end, with the API faked out."""
from datetime import datetime, timedelta, timezone

import pytest
from streamlit.testing.v1 import AppTest

import api

TIMEOUT = 30
NOW = datetime.now(timezone.utc)

# A customer is addressed by UUID and shown by account number. The ids here are
# stand-ins so a test cannot mix the two up.
NAMES = {"id-111": "Maya Haddad", "id-222": "Omar Fares"}
ACCOUNTS = {"id-111": "111", "id-222": "222"}


def _history() -> list[dict]:
    return [
        {
            "month": number, "label": f"Month {number}", "payment_status": 1,
            "payment_status_reading": "1 month late", "bill_amount": 1000.0,
            "amount_paid": 100.0, "utilisation": 0.8,
        }
        for number in range(1, 7)
    ]


def _detail(customer_id: str) -> dict:
    return {
        "customer_id": customer_id, "account_number": ACCOUNTS[customer_id],
        "full_name": NAMES[customer_id],
        "risk_band": "High", "probability": 0.7, "flagged": True,
        "credit_limit": 5000.0, "current_balance": 4000.0, "last_payment": 100.0,
        "months_in_arrears": 6, "segment": {"label": 1, "name": "Stretched"},
        "history": _history(), "latest_prediction": None,
        "highlights": {
            "utilisation_now": 0.8, "utilisation_mean": 0.8,
            "trajectory": "steady", "months_since_full_payment": None,
            "months_with_no_payment": 0,
        },
        "reasons": [
            {"feature": "PAY_1", "effect": 0.6, "description": "Last month was late"}
        ],
    }


def _requests() -> list[dict]:
    return [
        {
            "request_id": "r1", "customer_id": "id-111", "account_number": "111",
            "full_name": "Maya Haddad",
            "requested_at": (NOW - timedelta(days=3)).isoformat(),
            "reason": "I paid in full last month.", "status": "pending",
        },
    ]


def _queue() -> dict:
    return {
        "threshold": 0.5, "returned": 0, "total": 0, "rows": [],
        "summary_all": {
            "in_queue": 10, "flagged": 3, "bands": {"High": 2, "Medium": 3, "Low": 5},
        },
    }


@pytest.fixture
def faked_api(monkeypatch):
    """Every call the screen makes, answered locally. Returns the decisions recorded."""
    recorded = []

    def decisions(customer_id):
        return []

    decisions.clear = lambda: None

    fakes = {
        "review_requests": lambda: _requests(),
        "customer_detail": _detail,
        "recourse": lambda customer_id, token: {
            "customer_id": customer_id, "current_band": "High",
            "target_band": "Medium", "reachable": True, "achieved_band": "Medium",
            "steps": [{"description": "Pay 500 this month"}], "note": "Reachable.",
        },
        "search_customers": lambda q, limit=10: [
            {
                "customer_id": f"id-{q}", "account_number": q,
                "full_name": NAMES[f"id-{q}"], "known": False,
            }
        ],
        "users": lambda role=None: [],
        "queue": lambda **filters: _queue(),
        "decisions": decisions,
        "decide": lambda *args: recorded.append(args),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(api, name, fake)
    return recorded


def _app() -> None:
    from views import officer

    officer.render("officer@example.com", "token")


def _start() -> AppTest:
    return AppTest.from_function(_app, default_timeout=TIMEOUT).run()


def _text(app: AppTest) -> str:
    return " ".join(
        [element.value for element in app.markdown]
        + [element.value for element in app.subheader]
    )


def test_landing_is_search_first_with_no_customer_and_no_assistant(faked_api):
    app = _start()

    assert not app.exception
    assert "officer_customer_id" not in app.session_state
    assert app.text_input(key="officer_lookup").placeholder == (
        "Search by Customer ID or Name..."
    )
    text = _text(app)
    assert "Maya Haddad" in text and "3 days" in text
    assert "This month's triage" in text
    assert "LOANA AI" not in text
    assert len(app.chat_input) == 0


def test_a_search_lists_matches_and_opens_nothing_until_one_is_picked(faked_api):
    app = _start()
    app.text_input(key="officer_lookup").set_value("111").run()

    assert not app.exception
    assert "officer_customer_id" not in app.session_state
    assert len(app.chat_input) == 0

    app.button(key="officer_pick_id-111").click().run()

    assert app.session_state["officer_customer_id"] == "id-111"
    assert "LOANA AI" in _text(app)
    assert "Financial Health Copilot" in _text(app)


def test_review_opens_the_customer_with_the_request_banner(faked_api):
    app = _start()

    app.button(key="review_r1").click().run()

    assert not app.exception
    assert app.session_state["officer_customer_id"] == "id-111"
    text = _text(app)
    assert "Review request" in text
    assert "What the model saw" in text and "What the customer says" in text
    assert "What would change the rating" in text
    assert "Pay 500 this month" in text
    assert "Record the outcome of this review" in text
    assert "LOANA AI" in text


def test_a_customer_link_in_the_address_never_opens_a_customer(faked_api):
    app = AppTest.from_function(_app, default_timeout=TIMEOUT)
    app.query_params["customer"] = "111"
    app.run()

    assert "officer_customer_id" not in app.session_state
    assert len(app.chat_input) == 0


def test_recording_a_review_outcome_needs_a_justification_and_names_the_customer(
    faked_api,
):
    app = _start()
    app.button(key="review_r1").click().run()

    app.button(key="officer_record_id-111").click().run()
    assert "justification is required" in app.error[0].value
    assert faked_api == []

    app.text_area(key="officer_note_id-111").set_value("Late transfer, confirmed.").run()
    app.radio(key="officer_action_id-111").set_value("monitor").run()
    app.button(key="officer_record_id-111").click().run()

    assert not app.exception
    assert faked_api == [
        ("id-111", "officer@example.com", "monitor", "Late transfer, confirmed.")
    ]
    message = app.success[0].value
    assert "Maya Haddad" in message and "Monitor" in message
    assert app.text_area(key="officer_note_id-111").value == ""


def test_the_assistant_says_it_is_not_connected_and_keeps_the_layout(faked_api, monkeypatch):
    def unreachable(customer_id, messages):
        raise api.ServiceUnavailable("down")

    monkeypatch.setattr(api, "officer_chat", unreachable)
    app = _start()
    app.button(key="review_r1").click().run()

    app.chat_input(key="officer_ask").set_value("Why flagged?").run()

    assert not app.exception
    assert "Assistant not connected yet." in [info.value for info in app.info]
    assert app.session_state["officer_chat_id-111"] == []
    assert "LOANA AI" in _text(app)


def test_chat_history_is_kept_per_customer(faked_api, monkeypatch):
    monkeypatch.setattr(
        api, "officer_chat",
        lambda customer_id, messages: {"reply": f"About {customer_id}", "note": ""},
    )
    app = _start()
    app.button(key="review_r1").click().run()
    app.chat_input(key="officer_ask").set_value("Why flagged?").run()

    assert [m["content"] for m in app.session_state["officer_chat_id-111"]] == [
        "Why flagged?", "About id-111",
    ]

    app.button(key="officer_close").click().run()
    app.text_input(key="officer_lookup").set_value("222").run()
    app.button(key="officer_pick_id-222").click().run()

    assert app.session_state["officer_customer_id"] == "id-222"
    assert app.session_state["officer_chat_id-222"] == []
    assert len(app.session_state["officer_chat_id-111"]) == 2
    assert "About id-111" not in _text(app)


def test_the_prediction_survives_closing_and_reopening_the_customer(faked_api):
    app = _start()
    app.button(key="review_r1").click().run()
    app.session_state["prediction_id-111"] = {"probability": 0.5}

    app.button(key="officer_close").click().run()
    assert app.session_state["officer_customer_id"] is None
    assert "This month's triage" in _text(app)

    app.text_input(key="officer_lookup").set_value("111").run()
    app.button(key="officer_pick_id-111").click().run()

    assert app.session_state["officer_customer_id"] == "id-111"
    assert app.session_state["prediction_id-111"] == {"probability": 0.5}
