"""The pure logic behind the officer's review screen."""
from datetime import datetime, timedelta, timezone

import pytest

import review

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


def _ago(**delta) -> str:
    return (NOW - timedelta(**delta)).isoformat()


def _request(request_id, customer_id="1", status="pending", **delta) -> dict:
    return {
        "request_id": request_id,
        "customer_id": customer_id,
        "account_number": f"acct-{customer_id}",
        "full_name": None,
        "requested_at": _ago(**delta),
        "reason": "Please look again.",
        "status": status,
    }


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ({"seconds": 20}, "just now"),
        ({"minutes": 1}, "1 minute"),
        ({"minutes": 45}, "45 minutes"),
        ({"hours": 1}, "1 hour"),
        ({"hours": 5, "minutes": 30}, "5 hours"),
        ({"days": 1}, "1 day"),
        ({"days": 3, "hours": 20}, "3 days"),
        ({"seconds": -90}, "just now"),
    ],
)
def test_waiting_text_uses_the_largest_whole_unit(delta, expected):
    assert review.waiting_text(_ago(**delta), NOW) == expected


def test_a_timestamp_without_a_zone_is_taken_as_utc():
    naive = (NOW - timedelta(days=2)).replace(tzinfo=None).isoformat()
    assert review.waiting_text(naive, NOW) == "2 days"


def test_truncate_cuts_at_a_word_and_marks_the_cut():
    text = "word " * 60
    cut = review.truncate(text, limit=20)
    assert cut == "word word word word…"
    assert review.truncate("short note") == "short note"


def test_open_requests_are_oldest_first_and_exclude_resolved():
    requests = [
        _request("new", days=1),
        _request("done", status="resolved", days=9),
        _request("old", status="in_review", days=5),
    ]
    assert [item["request_id"] for item in review.open_requests(requests)] == [
        "old", "new",
    ]
    assert review.pending_count(requests) == 1


def test_request_for_finds_the_customers_open_request_only():
    requests = [
        _request("a", customer_id="1", status="resolved", days=8),
        _request("b", customer_id="1", days=2),
        _request("c", customer_id="2", days=1),
    ]
    assert review.request_for(requests, "1")["request_id"] == "b"
    assert review.request_for(requests, "3") is None
    assert review.request_for(requests, None) is None


def test_card_data_names_an_unnamed_customer_by_account_and_flags_long_notes():
    long_note = "I paid late once. " * 20
    item = {
        **_request("a", customer_id="uuid-1", days=3),
        "account_number": "14861",
        "reason": long_note,
    }
    card = review.card_data(item, "High", NOW)

    assert card["name"] == "Customer 14861"
    assert card["account"] == "14861"
    assert card["waiting"] == "3 days"
    assert card["status"] == "Pending"
    assert card["band"] == "High"
    assert card["is_truncated"] and card["note"].endswith("…")
    assert card["full_note"] == long_note.strip()


def test_card_data_leaves_a_short_note_whole():
    card = review.card_data(
        {**_request("a"), "full_name": "Maya Haddad"}, None, NOW
    )
    assert card["name"] == "Maya Haddad"
    assert not card["is_truncated"]


def _customers(count: int) -> list[dict]:
    return [
        {
            "customer_id": f"uuid-{number}",
            "account_number": str(number),
            "full_name": "Maya Haddad" if number == 2 else None,
        }
        for number in range(1, count + 1)
    ]


def test_demo_requests_are_deterministic_and_use_the_given_customers():
    first = review.demo_requests(_customers(3), NOW)
    again = review.demo_requests(_customers(3), NOW)

    assert first == again
    assert [item["customer_id"] for item in first] == ["uuid-1", "uuid-2", "uuid-3"]
    assert [item["account_number"] for item in first] == ["1", "2", "3"]
    assert first[1]["full_name"] == "Maya Haddad" and first[0]["full_name"] is None
    assert {item["status"] for item in first} <= {"pending", "in_review", "resolved"}
    assert len({item["request_id"] for item in first}) == 3


def test_demo_requests_never_outnumber_the_customers():
    assert review.demo_requests([], NOW) == []
    assert len(review.demo_requests(_customers(1), NOW)) == 1


def _detail(**overrides) -> dict:
    history = [
        {"bill_amount": 1000.0, "amount_paid": 400.0},
        {"bill_amount": 1000.0, "amount_paid": 0.0},
    ]
    detail = {
        "months_in_arrears": 2,
        "history": history,
        "highlights": {
            "months_with_no_payment": 1,
            "utilisation_now": 0.85,
            "utilisation_mean": 0.6,
            "trajectory": "deteriorating",
        },
    }
    detail.update(overrides)
    return detail


def test_model_saw_reads_the_history_in_plain_sentences():
    sentences = review.model_saw(_detail())

    assert "Behind on payments in 2 of the last 6 months." in sentences
    assert "No payment at all in 1 month." in sentences
    assert "Using 85% of the limit now, 60% on average over six months." in sentences
    assert "Paid 400 against 2,000 billed (20%)." in sentences
    assert "Repayment behaviour is deteriorating." in sentences


def test_model_saw_copes_with_a_clean_account_and_no_bills():
    clean = _detail(
        months_in_arrears=0,
        history=[{"bill_amount": 0.0, "amount_paid": 0.0}],
        highlights={"months_with_no_payment": 0},
    )
    assert review.model_saw(clean) == ["Never behind on a payment in the last six months."]


REASONS = [
    {"feature": "fixed_feature", "effect": 0.9, "description": "Something fixed"},
    {"feature": "PAY_1", "effect": 0.5, "description": "Last month was late"},
    {"feature": "PAY_AMT1", "effect": -0.3, "description": "Paid a lot"},
]


def test_unchangeable_reason_is_reported_when_the_top_reason_cannot_be_acted_on():
    assert review.unchangeable_reason(REASONS, {"PAY_1"}) == "Something fixed"


def test_unchangeable_reason_is_silent_when_the_top_reason_is_actionable():
    assert review.unchangeable_reason(REASONS[1:], {"PAY_1"}) is None
    assert review.unchangeable_reason(REASONS[2:], {"PAY_1"}) is None
