"""My Transactions: real payments, one account type, and the drawer entry."""
from streamlit.testing.v1 import AppTest

from ui import nav, transactions

TIMEOUT = 30
HISTORY = [
    {"month": month, "label": f"M{month}", "amount_paid": paid, "bill_amount": bill}
    for month, paid, bill in ((1, 3000, 4000), (2, 800, 6000), (3, 0, 5000), (4, 2000, 7000), (5, 1500, 8000), (6, 0, 9000))
]


def test_rows_run_newest_first_with_month_labels():
    rows = transactions.rows_for("customer-a", HISTORY)
    assert [row["Amount"] for row in rows] == ["94.25", "25.13", "62.83", "47.13"]
    assert [row["Month"] for row in rows] == ["M1", "M2", "M4", "M5"]
    assert len({row["Account ID"] for row in rows}) == 1
    assert len({row["Reference"] for row in rows}) == 4
    assert "Unpaid principal" not in rows[0]


def test_rows_are_stable_between_calls():
    assert transactions.rows_for("c", HISTORY) == transactions.rows_for("c", HISTORY)


def test_fake_card_is_masked():
    masked, card_id = transactions.card_for("customer-a")
    assert masked.startswith("•••• •••• •••• ") and card_id.startswith("CC-")


def _app() -> None:
    from ui import drawer, nav

    drawer.render("Omar Fares", "Cardholder", lambda: None, items=nav.ACCOUNT_ITEMS)


def test_drawer_lists_my_transactions_after_the_overview():
    app = AppTest.from_function(_app, default_timeout=TIMEOUT).run()
    app.button(key="nav_toggle").click().run()
    keys = [button.key for button in app.button if button.key.startswith("nav_")]
    assert keys.index(f"nav_{nav.TRANSACTIONS}") == keys.index("nav_dashboard") + 1
