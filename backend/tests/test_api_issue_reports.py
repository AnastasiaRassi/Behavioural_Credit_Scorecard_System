"""Anyone signed in can report an issue; it is stored against them."""
from app.db.models import IssueReport, User
from tests.conftest import bearer
from tests.test_api import OFFICER_EMAIL, api_session, client  # noqa: F401  (fixture imports)

URL = "/issue-reports"
KIND = "Feedback or idea"


def test_a_signed_in_person_can_report_and_it_is_stored(client, api_session):
    sent = client.post(
        URL, json={"kind": KIND, "details": "  The chart is hard to read.  "},
        headers=bearer(OFFICER_EMAIL),
    )
    assert sent.status_code == 201

    officer = api_session.query(User).filter_by(email=OFFICER_EMAIL).one()
    row = api_session.query(IssueReport).filter_by(user_id=officer.id).one()
    assert (row.kind, row.details) == (KIND, "The chart is hard to read.")


def test_an_anonymous_report_is_refused(client):
    assert client.post(URL, json={"kind": KIND, "details": "Hello"}).status_code == 401


def test_a_blank_or_unknown_report_is_rejected(client):
    headers = bearer(OFFICER_EMAIL)
    assert client.post(URL, json={"kind": KIND, "details": "   "}, headers=headers).status_code == 422
    assert client.post(URL, json={"kind": "Other", "details": "x"}, headers=headers).status_code == 422
