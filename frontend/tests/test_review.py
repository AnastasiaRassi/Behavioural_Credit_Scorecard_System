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


def test_conversations_group_open_messages_per_customer():
    requests = [
        _request("a", "1", "pending", days=3) | {"reason": "First."},
        _request("b", "1", "pending", days=1) | {"reason": "Second."},
        _request("c", "2", "pending", days=2),
        _request("d", "3", "resolved", days=5),
    ]

    grouped = review.conversations(requests)

    assert [item["customer_id"] for item in grouped] == ["1", "2"]
    assert grouped[0]["request_id"] == "a"
    assert grouped[0]["unread"] == 2 and grouped[0]["latest"] == "Second."
    assert review.unread_by_customer(requests) == {"1": 2, "2": 1}


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


def test_capped_counts_stop_at_the_cap():
    assert review.capped(5, 99) == "5"
    assert review.capped(99, 99) == "99"
    assert review.capped(5000, 99) == "99+"


def _entry(kind, status=None) -> dict:
    return {"kind": kind, "status": status}


@pytest.mark.parametrize("entries, stage", [
    ([], None),
    ([_entry("message", "pending")], 1),
    ([_entry("message", "in_review")], 2),
    ([_entry("message", "resolved"), _entry("decision")], 3),
    ([_entry("message", "resolved"), _entry("decision"), _entry("message", "pending")], 1),
])
def test_case_stage_follows_the_newest_request(entries, stage):
    assert review.case_stage(entries) == stage


def _account(behind: int, latest: int, flagged: bool = True) -> dict:
    history = [
        {"payment_status": latest if i == 0 else 0, "bill_amount": 120_000, "amount_paid": 10_000}
        for i in range(6)
    ]
    return {"needs_attention": flagged, "risk_band": "High", "months_in_arrears": behind,
            "history": history}


def test_banner_wording_follows_how_far_behind_the_account_is():
    from views.cardholder import banner_copy

    tone, title, body = banner_copy(_account(6, 2))
    assert title == "Your account is seriously behind"
    assert "6 of the last 6 months" in body and "early heads-up" not in body
    assert "8%" in body  # payments covered a small share of the bills

    assert banner_copy(_account(1, 1))[1] == "Your account has started to fall behind"
    assert banner_copy(_account(0, 0))[1] == "Your next payment may be at risk"
    assert banner_copy(_account(0, 0, flagged=False))[0] == "good"
