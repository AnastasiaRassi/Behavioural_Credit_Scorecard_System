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


def _briefing() -> dict:
    lead = _cohort({})["rows"][0]
    return {
        "text": "Not scored yet: **41** getting worse and 2+ months behind.",
        "cohorts": [
            {"label": "Getting worse, 2+ months behind",
             "filter": {"min_months_in_arrears": 2, "trajectory": "deteriorating",
                        "not_scored": True},
             "total": 41},
            {"label": "No payment in 3+ months",
             "filter": {"min_months_with_no_payment": 3, "not_scored": True}, "total": 0},
        ],
        "lead": lead,
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
        "briefing": lambda: _briefing(),
        "triage": lambda messages, asked_by=None: {
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

    officer.render("officer@example.com", "token")


def _start() -> AppTest:
    return AppTest.from_function(_app, default_timeout=TIMEOUT).run()


def _show(app: AppTest, view: str) -> None:
    """What clicking a drawer entry does."""
    app.session_state["active_view"] = view
    app.run()


def _open_chat(app: AppTest) -> None:
    app.button(key="loana_launch_button").click().run()


def _text(app: AppTest) -> str:
    return " ".join(
        [element.value for element in app.markdown]
        + [element.value for element in app.subheader]
    )


def _customer_chat(app: AppTest) -> list:
    return [box for box in app.chat_input if box.key == "officer_ask"]


def _command(app: AppTest, text: str) -> None:
    app.chat_input(key="officer_command").set_value(text).run()


def test_landing_opens_with_loanas_briefing_and_one_command_bar(faked_api):
    app = _start()

    assert not app.exception
    assert "officer_customer_id" not in app.session_state
    assert app.chat_input(key="officer_command").placeholder.startswith("Ask LOANA")
    text = _text(app)
    assert "Worklist" in text and "Triage assistant" in text
    assert "Good " in text
    assert "Customers asking for a human review" not in text
    assert "No customers pulled in yet" in " ".join(i.value for i in app.info)
    # The briefing names who to start with and counts each suggestion.
    assert app.button(key="officer_lead").label == "Open Omar Fares's dashboard →"
    assert app.button(key="officer_quick_0").label.endswith("· 41")
    assert app.button(key="officer_quick_1").disabled
    assert _customer_chat(app) == []


def test_the_lead_opens_their_dashboard(faked_api):
    app = _start()
    app.button(key="officer_lead").click().run()

    assert app.session_state["officer_customer_id"] == "id-222"


def test_a_suggestion_fills_the_worklist_with_its_filter_as_chips(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()

    assert not app.exception
    page = app.session_state["officer_cohort"]
    assert page["total"] == 41 and page["rows"][0]["account_number"] == "222"
    assert "Showing 1 of 41" in " ".join(c.value for c in app.caption)
    assert app.button(key="officer_predict_selected").disabled
    chips = app.button_group[0]
    assert set(chips.value) == {"min_months_in_arrears", "trajectory", "not_scored"}


def test_dropping_a_chip_widens_the_filter(faked_api, monkeypatch):
    asked = []
    monkeypatch.setattr(api, "cohort", lambda criteria: asked.append(criteria) or _cohort(criteria))
    app = _start()
    app.button(key="officer_quick_0").click().run()
    app.button_group[0].set_value(["min_months_in_arrears", "not_scored"]).run()

    assert not app.exception
    assert "trajectory" not in asked[-1]
    assert asked[-1]["min_months_in_arrears"] == 2


def test_asking_loana_fills_the_worklist_and_says_how_many_match(faked_api):
    app = _start()
    _command(app, "2+ months behind")

    assert not app.exception
    assert app.session_state["officer_cohort"]["total"] == 41
    reply = app.session_state["officer_triage_chat"][-1]["content"]
    assert "41 match" in reply and "first 1" in reply


def test_every_row_opens_the_user_dashboard(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()
    button = app.button(key="officer_open_id-222")

    assert button.label == "View user dashboard →"
    button.click().run()
    assert app.session_state["officer_customer_id"] == "id-222"


def test_ticking_a_row_enables_predict_selected(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()
    tick = next(box for box in app.checkbox if box.key.endswith("_id-222"))
    tick.check().run()

    assert app.button(key="officer_predict_selected").label == "Predict selected (1)"
    assert not app.button(key="officer_predict_selected").disabled


def test_a_just_scored_customer_opens_their_analysis_and_the_worklist_survives(faked_api):
    app = _start()
    app.button(key="officer_quick_0").click().run()
    app.session_state["officer_just_scored"] = [
        {"customer_id": "id-222", "label": "Omar Fares · 222", "risk_band": "High",
         "flagged": True}
    ]
    app.run()

    assert "Scored 1" in _text(app)
    app.button(key="officer_scored_id-222").click().run()

    assert not app.exception
    assert app.session_state["officer_customer_id"] == "id-222"

    _show(app, "dashboard")

    assert app.session_state["officer_cohort"]["total"] == 41
    assert app.button(key="officer_scored_id-222") is not None


def test_a_search_lists_matches_and_opens_nothing_until_one_is_picked(faked_api):
    app = _start()
    _command(app, "111")

    assert not app.exception
    assert "officer_customer_id" not in app.session_state
    assert "officer_triage_chat" not in app.session_state
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


def test_a_group_request_goes_to_loana_not_search(faked_api, monkeypatch):
    searched = []
    monkeypatch.setattr(api, "search_customers", lambda q, limit=10: searched.append(q) or [])
    app = _start()
    _command(app, "customers getting worse")

    assert searched == []
    assert app.session_state["officer_cohort"]["total"] == 41


def test_review_opens_the_customer_with_the_unread_alert_and_thread(faked_api):
    app = _start()
    _show(app, "chat")

    app.button(key="review_r1").click().run()

    assert not app.exception
    assert app.session_state["officer_conversation_id"] == "id-111"
    text = _text(app)
    # The same chat card the cardholder sees, with a way back to the worklist.
    assert app.button(key="officer_conversation_dashboard").label == "Maya Haddad"
    assert "Submitted" in text
    assert app.button(key="back_to_dashboard").label.endswith("Back to worklist")
    assert app.button(key="officer_record_id-111").label == "Issue decision"

    app.button(key="officer_conversation_dashboard").click().run()
    text = _text(app)
    assert "1 unanswered request from this customer" in text
    assert "What the model saw" in text and "What would change the rating" in text
    assert "Pay 500 this month" in text


def test_the_dashboard_links_to_messages_instead_of_a_chat_box(faked_api):
    app = _start()
    _command(app, "222")
    app.button(key="officer_pick_id-222").click().run()

    assert "officer_note_id-222" not in [area.key for area in app.text_area]
    app.button(key="officer_message_id-222").click().run()

    assert app.session_state["officer_conversation_id"] == "id-222"
    assert "No review requests yet" in _text(app)


def test_back_to_worklist_returns_to_the_landing(faked_api):
    app = _start()
    _show(app, "chat")
    app.button(key="review_r1").click().run()
    app.button(key="back_to_dashboard").click().run()

    assert app.chat_input(key="officer_command") is not None


def test_open_conversation_opens_it_in_the_drawer_not_as_a_customer(faked_api):
    app = _start()
    _show(app, "chat")

    app.button(key=[b.key for b in app.button if b.key.startswith("review_")][0]).click().run()

    assert app.session_state["officer_conversation_id"] == "id-111"
    assert "officer_customer_id" not in app.session_state or not app.session_state["officer_customer_id"]


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
    _show(app, "chat")
    app.button(key="review_r1").click().run()

    app.button(key="officer_record_id-111").click().run()
    assert "note is required" in app.error[0].value
    assert faked_api == []

    app.text_area(key="officer_note_id-111").set_value("Late transfer, confirmed.").run()
    app.button(key="officer_record_id-111").click().run()

    assert not app.exception
    assert faked_api == [
        ("id-111", "officer@example.com", "no_action", "Late transfer, confirmed.")
    ]
    message = app.success[0].value
    assert "Maya Haddad" in message
    assert "review closed" in message
    assert app.text_area(key="officer_note_id-111").value == ""


def test_attaching_stats_appends_them_to_the_reply_without_the_probability(faked_api):
    app = _start()
    _show(app, "chat")
    app.button(key="review_r1").click().run()

    app.text_area(key="officer_note_id-111").set_value("Hello").run()
    app.button(key="officer_attach_all_id-111").click().run()

    assert not app.exception
    note = app.text_area(key="officer_note_id-111").value
    assert note.startswith("Hello\n") and "Credit limit:" in note
    assert "robab" not in note


def test_the_assistant_says_it_is_not_connected_and_keeps_the_layout(faked_api, monkeypatch):
    def unreachable(customer_id, messages):
        raise api.ServiceUnavailable("down")

    monkeypatch.setattr(api, "officer_chat", unreachable)
    app = _start()
    _show(app, "chat")
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
    _show(app, "chat")
    app.button(key="review_r1").click().run()
    _open_chat(app)
    app.chat_input(key="officer_ask").set_value("Why flagged?").run()

    assert [m["content"] for m in app.session_state["officer_chat_id-111"]] == [
        "Why flagged?", "About id-111",
    ]

    _show(app, "dashboard")
    _command(app, "222")
    app.button(key="officer_pick_id-222").click().run()

    assert app.session_state["officer_customer_id"] == "id-222"
    assert app.session_state["officer_chat_id-222"] == []
    assert len(app.session_state["officer_chat_id-111"]) == 2
    assert "About id-111" not in _text(app)


def test_the_prediction_survives_closing_and_reopening_the_customer(faked_api):
    app = _start()
    _show(app, "chat")
    app.button(key="review_r1").click().run()
    app.session_state["prediction_id-111"] = {"probability": 0.5}

    _show(app, "dashboard")
    assert "Worklist" in _text(app)

    _command(app, "111")
    app.button(key="officer_pick_id-111").click().run()

    assert app.session_state["officer_customer_id"] == "id-111"
    assert app.session_state["prediction_id-111"] == {"probability": 0.5}


def test_a_large_backlog_is_capped_in_messages(faked_api, monkeypatch):
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
    _show(app, "chat")

    assert not app.exception
    assert "5000 waiting" in " ".join(c.value for c in app.caption)
    assert len([b for b in app.button if b.key.startswith("review_")]) == 10
    assert "4,990 more are waiting" in " ".join(c.value for c in app.caption)
