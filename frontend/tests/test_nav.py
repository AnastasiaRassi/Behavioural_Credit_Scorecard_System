"""Drawer navigation: the sidebar stays open while views switch."""
from streamlit.testing.v1 import AppTest

TIMEOUT = 30


def _app() -> None:
    import streamlit as st

    from ui import drawer, nav

    drawer.render("Omar Fares", "Cardholder", lambda: None, items=nav.ACCOUNT_ITEMS)
    st.write(f"view={nav.current()}")


def _text(app: AppTest) -> str:
    return " ".join(element.value for element in app.markdown)


def test_drawer_switches_views_and_stays_open():
    app = AppTest.from_function(_app, default_timeout=TIMEOUT).run()
    assert "Account Overview" not in _text(app)

    app.button(key="nav_toggle").click().run()
    assert app.button(key="nav_chat") is not None

    app.button(key="nav_chat").click().run()
    assert "view=chat" in _text(app)
    app.button(key="nav_dashboard").click().run()
    assert "view=dashboard" in _text(app)


def test_profile_item_opens_the_profile_view():
    app = AppTest.from_function(_app, default_timeout=TIMEOUT).run()
    app.button(key="nav_toggle").click().run()
    app.button(key="nav_profile").click().run()
    assert "view=profile" in _text(app)


def _officer_app() -> None:
    from ui import drawer, nav

    drawer.render("Sam Lee", "Loan officer", lambda: None, nav.OFFICER_ITEMS, "Omar Fares")


def test_officer_drawer_indents_the_opened_customer_under_filter_users():
    app = AppTest.from_function(_officer_app, default_timeout=TIMEOUT).run()
    app.button(key="nav_toggle").click().run()

    keys = [button.key for button in app.button if button.key.startswith("nav_")]
    assert keys.index("nav_customer") == keys.index("nav_dashboard") + 1
    assert "Filter Users" in app.button(key="nav_dashboard").label


def test_closing_the_customer_block_removes_it_and_leaves_its_page():
    def app_fn() -> None:
        import streamlit as st

        from ui import drawer, nav

        def close() -> None:
            st.session_state["label"] = None
            nav.go(nav.DASHBOARD)

        drawer.render(
            "Sam Lee", "Loan officer", lambda: None, nav.OFFICER_ITEMS,
            st.session_state.setdefault("label", "Omar Fares"), close,
        )
        st.write(f"view={nav.current()}")

    app = AppTest.from_function(app_fn, default_timeout=TIMEOUT).run()
    app.button(key="nav_toggle").click().run()
    app.button(key="nav_customer").click().run()
    assert "view=customer" in _text(app)

    app.button(key="nav_customer_close").click().run()
    assert "view=dashboard" in _text(app)
    assert "nav_customer" not in [button.key for button in app.button]
