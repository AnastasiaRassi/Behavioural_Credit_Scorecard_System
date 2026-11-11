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


def _cohort(criteria: dict) -> dict:
    return {
        "description": "Customers with 2+ months in arrears, most months in arrears first.",
        "total": 41,
        "rows": [
            {
                "customer_id": "id-222", "account_number": "222", "full_name": "Omar Fares",
                "credit_limit": 5000.0, "current_balance": 4000.0, "last_payment": 0.0,
                "months_in_arrears": 4, "utilisation_now": 0.8,
                "trajectory": "deteriorating", "months_with_no_payment": 3,
                "last_risk_band": None, "last_scored_at": None,
            }
        ],
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
        "cohort": _cohort,
        "triage": lambda messages: {
            "reply": "Customers with 2+ months in arrears, most months in arrears first.",
            "source": "model", "note": "",
            "filter": {"min_months_in_arrears": 2}, "cohort": _cohort({}),
        },
        "decisions": decisions,
        "thread": lambda customer_id: [
            {
                "id": "r1", "kind": "message",
                "at": (NOW - timedelta(days=3)).isoformat(), "author": "Maya Haddad",
                "body": "I paid in full last month.", "action": None, "status": "pending",
            }
        ] if customer_id == "id-111" else [],
        "decide": lambda *args: recorded.append(args),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(api, name, fake)
    return recorded


def _app() -> None:
    from views import officer

    officer.render("officer@example.com", "token", officer.back_button)


def _start() -> AppTest:
    return AppTest.from_function(_app, default_timeout=TIMEOUT).run()


def _open_chat(app: AppTest) -> None:
    app.button(key="loana_launch_button").click().run()


def _text(app: AppTest) -> str:
    return " ".join(
        [element.value for element in app.markdown]
        + [element.value for element in app.subheader]
    )


def _customer_chat(app: AppTest) -> list:
    return [box for box in app.chat_input if box.key == "officer_ask"]


def test_landing_is_search_requests_and_an_empty_worklist_beside_loana(faked_api):
    app = _start()

    assert not app.exception
    assert "officer_customer_id" not in app.session_state
    assert app.text_input(key="officer_lookup").placeholder == (
        "Find a customer by ID or name..."
    )
    text = _text(app)
    assert "Maya Haddad" in text and "3 days" in text
    assert "1 customer is waiting for a reply" in text
    assert app.button(key="inbox_id-111") is not None
    assert "Worklist" in text and "Triage assistant" in text
    assert "No customers pulled in yet" in " ".join(i.value for i in app.info)
    # Loana here pulls cohorts; the per-customer chat only exists on a record.
    assert app.chat_input(key="officer_triage_ask") is not None
    assert _customer_chat(app) == []


def test_a_quick_cohort_fills_the_worklist_unscored(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()

    assert not app.exception
    page = app.session_state["officer_cohort"]
    assert page["total"] == 41 and page["rows"][0]["account_number"] == "222"
    assert "Showing 1 of 41" in " ".join(c.value for c in app.caption)
    assert app.button(key="officer_predict_selected").disabled


def test_asking_loana_fills_the_worklist_and_says_how_many_match(faked_api):
    app = _start()
    app.chat_input(key="officer_triage_ask").set_value("2+ months behind").run()

    assert not app.exception
    assert app.session_state["officer_cohort"]["total"] == 41
    reply = app.session_state["officer_triage_chat"][-1]["content"]
    assert "41 match" in reply and "first 1" in reply


def test_a_just_scored_customer_opens_their_analysis_and_the_worklist_survives(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()
    # What "predict selected" leaves behind; the table's ticks cannot be set in AppTest.
    app.session_state["officer_just_scored"] = [
        {"customer_id": "id-222", "label": "Omar Fares · 222", "risk_band": "High",
         "flagged": True}
    ]
    app.run()

    assert "Scored 1" in _text(app)
    app.button(key="officer_scored_id-222").click().run()

    assert not app.exception
    assert app.session_state["officer_customer_id"] == "id-222"

    app.button(key="officer_close").click().run()

    assert app.session_state["officer_customer_id"] is None
    assert app.session_state["officer_cohort"]["total"] == 41
    assert app.button(key="officer_scored_id-222") is not None


def test_view_analysis_waits_for_exactly_one_ticked_customer(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()

    assert app.button(key="officer_view_analysis").disabled


def test_a_search_lists_matches_and_opens_nothing_until_one_is_picked(faked_api):
    app = _start()
    app.text_input(key="officer_lookup").set_value("111").run()

    assert not app.exception
    assert "officer_customer_id" not in app.session_state
    assert _customer_chat(app) == []

    app.button(key="officer_pick_id-111").click().run()

    assert app.session_state["officer_customer_id"] == "id-111"
    assert "LOANA AI" not in _text(app)
    assert len(app.chat_input) == 0

    _open_chat(app)

    assert "LOANA AI" in _text(app)
    assert "Financial Health Copilot" in _text(app)

    app.button(key="loana_close").click().run()

    assert "LOANA AI" not in _text(app)
    assert len(app.chat_input) == 0


def test_review_opens_the_customer_with_the_unread_alert_and_thread(faked_api):
    app = _start()

    app.button(key="review_r1").click().run()

    assert not app.exception
    assert app.session_state["officer_customer_id"] == "id-111"
    text = _text(app)
    assert "1 unanswered message from this customer" in text
    assert "What the model saw" in text and "What the customer says" in text
    assert "What would change the rating" in text
    assert "Pay 500 this month" in text
    assert "Activity &amp; communication history" in text or (
        "Activity & communication history" in text
    )
    assert "Awaiting reply" in text
    assert "Reply with a decision" in text
    assert "Back to worklist" in app.button(key="officer_close").label


def test_the_inbox_shortcut_opens_the_waiting_customer(faked_api):
    app = _start()

    app.button(key="inbox_id-111").click().run()

    assert app.session_state["officer_customer_id"] == "id-111"


def test_a_customer_link_in_the_address_never_opens_a_customer(faked_api):
    app = AppTest.from_function(_app, default_timeout=TIMEOUT)
    app.query_params["customer"] = "111"
    app.run()

    assert "officer_customer_id" not in app.session_state
    assert _customer_chat(app) == []


def test_recording_a_review_outcome_needs_a_justification_and_names_the_customer(
    faked_api,
):
    app = _start()
    app.button(key="review_r1").click().run()

    app.button(key="officer_record_id-111").click().run()
    assert "note is required" in app.error[0].value
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
    assert "conversation closed" in message
    assert app.text_area(key="officer_note_id-111").value == ""


def test_the_assistant_says_it_is_not_connected_and_keeps_the_layout(faked_api, monkeypatch):
    def unreachable(customer_id, messages):
        raise api.ServiceUnavailable("down")

    monkeypatch.setattr(api, "officer_chat", unreachable)
    app = _start()
    app.button(key="review_r1").click().run()
    _open_chat(app)

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
    _open_chat(app)
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
    assert "Worklist" in _text(app)

    app.text_input(key="officer_lookup").set_value("111").run()
    app.button(key="officer_pick_id-111").click().run()

    assert app.session_state["officer_customer_id"] == "id-111"
    assert app.session_state["prediction_id-111"] == {"probability": 0.5}


def test_a_large_backlog_is_capped_on_the_landing(faked_api, monkeypatch):
    backlog = [
        {
            "request_id": f"r{number}", "customer_id": f"id-{number}",
            "account_number": str(number), "full_name": f"Customer {number}",
            "requested_at": (NOW - timedelta(minutes=number)).isoformat(),
            "reason": "Please look.", "status": "pending",
        }
        for number in range(5000)
    ]
    monkeypatch.setattr(api, "review_requests", lambda: backlog)
    monkeypatch.setattr(api, "customer_detail", lambda customer_id: {"latest_prediction": None})
    app = _start()

    assert not app.exception
    text = _text(app)
    assert ">99+</span>" in text and "5,000 customers are waiting" in text
    assert len([b for b in app.button if b.key.startswith("review_")]) == 10
    assert "4,990 more are waiting" in " ".join(c.value for c in app.caption)
