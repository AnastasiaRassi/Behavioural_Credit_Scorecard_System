"""HTTP client for the backend. The frontend reads nothing from disk.

Reading `artifacts/` directly would let a screen hold data its role is not
entitled to and merely decline to draw it: the cardholder's probability would
still be one devtools panel away. The API decides what each role is sent, so the
frontend asks for it.

Shapes are adapted here rather than in the tabs, so the screens keep one vocabulary
whichever endpoint feeds them.
"""
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

import streamlit as st

logger = logging.getLogger(__name__)


def _log_to_file() -> None:
    """Mirror this module's warnings into the shared log dir, so the dashboard's side
    of a failure sits next to the API's instead of only in the container's stdout."""
    if any(getattr(h, "_frontend_file", False) for h in logger.handlers):
        return
    folder = "testing" if os.environ.get("APP_ENV") == "test" else "app"
    path = Path(__file__).resolve().parents[1] / "backend" / "logs" / folder
    try:
        path.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path / "frontend.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    except OSError:
        return  # a read-only mount must not take the dashboard down
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler._frontend_file = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_log_to_file()

# The only thing a screen says when the service fails, whatever the cause.
UNREACHABLE = "CrediTrait can't connect right now. Try again in a moment."

# Overridable so the dashboard can be pointed at an API on another port without
# editing it, which is also what lets two instances run side by side.
BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8014")
TIMEOUT_SECONDS = 30

START_COMMAND = "docker compose up --build"

# Recorded baseline. The status light goes amber if test AUC drifts from this.
BASELINE_AUC = 0.781
AUC_TOLERANCE = 0.02
FOUR_FIFTHS = 0.80


class ServiceUnavailable(RuntimeError):
    """A failure of the service, not of the person using it.

    `str(...)` is the one sentence a screen may show. The status code, the URL
    and the body stay in `technical`, which goes to the log: a route that moved
    or a process that died is an operator's problem, and putting FastAPI's JSON
    in front of a loan officer tells them nothing they can act on.
    """

    def __init__(
        self,
        technical: str,
        status: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
    ):
        # The code and request id are what a person can quote and an engineer can
        # search for in the API log; neither says anything about the cause.
        reference = " · ".join(part for part in (request_id, code) if part)
        super().__init__(f"{UNREACHABLE} (ref {reference})" if reference else UNREACHABLE)
        # Carried so a caller can tell a rejected credential from a dead service
        # without parsing the message.
        self.status = status
        self.technical = technical
        self.code = code
        self.request_id = request_id


class SignInRejected(RuntimeError):
    pass


def _request(
    path: str,
    payload: dict | None = None,
    method: str | None = None,
    token: str | None = None,
    timeout: float = TIMEOUT_SECONDS,
):
    url = BASE_URL + path
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exception:
        detail = exception.read().decode()[:400]
        code, request_id = _error_reference(detail, exception.headers)
        technical = f"{exception.code} {code or ''} ref={request_id or '-'} from {method or 'GET'} {url}: {detail}"
        logger.warning("%s", technical)
        raise ServiceUnavailable(
            technical, status=exception.code, code=code, request_id=request_id
        ) from exception
    except TimeoutError as exception:
        # A read that outlives the timeout raises this bare, not wrapped in URLError.
        # The chat turn is the call most likely to, so it must not take the page down.
        technical = f"no reply from {method or 'GET'} {url} within {timeout}s"
        logger.warning("%s", technical)
        raise ServiceUnavailable(technical, code="API_TIMEOUT") from exception
    except urllib.error.URLError as exception:
        technical = (
            f"cannot reach the API at {BASE_URL} ({exception}). "
            f"Start it with: {START_COMMAND}"
        )
        logger.warning("%s", technical)
        raise ServiceUnavailable(technical, code="API_UNREACHABLE") from exception


def _error_reference(body: str, headers) -> tuple[str | None, str | None]:
    """The API's error code and request id: from its JSON body, else its header."""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = {}
    parsed = parsed if isinstance(parsed, dict) else {}
    return parsed.get("code"), parsed.get("request_id") or headers.get("X-Request-ID")


def _get(path: str, **params):
    supplied = {k: v for k, v in params.items() if v is not None}
    query = f"?{urllib.parse.urlencode(supplied)}" if supplied else ""
    return _request(path + query)


def _post(path: str, payload: dict | None = None, **params):
    supplied = {k: v for k, v in params.items() if v is not None}
    query = f"?{urllib.parse.urlencode(supplied)}" if supplied else ""
    return _request(path + query, payload=payload or {})


def guard(exception: ServiceUnavailable) -> None:
    """Show a failure the user can act on, then stop the script."""
    st.error(str(exception))
    st.stop()


# --- signing in -----------------------------------------------------------


def sign_in(email: str, password: str) -> dict:
    """Exchange credentials for a session: the token, and who the caller is.

    Never cached. A sign-in is not a read, and caching one would hold a
    credential answer longer than the attempt it belongs to.
    """
    try:
        return _request("/session", payload={"email": email, "password": password})
    except ServiceUnavailable as exception:
        if exception.status == 401:
            raise SignInRejected("Email or password is incorrect") from exception
        raise


def resume(token: str) -> dict | None:
    """Rebuild a session from a stored token, or None if the API no longer accepts it."""
    try:
        account = _request("/session", token=token)
    except ServiceUnavailable as exception:
        if exception.status == 401:
            return None
        raise
    return {**account, "token": token}


# --- roles ----------------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def roles() -> list[dict]:
    return _get("/roles")


@st.cache_data(ttl=60, show_spinner=False)
def users(role: str | None = None) -> list[dict]:
    return _get("/users", role=role)


# --- ML engineer ----------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def config() -> dict:
    """The model report, keyed as the engineer tabs already expect."""
    report = dict(_get("/report"))
    # /report publishes the operating point as `threshold`; the tabs grew up on
    # the artifact's name for it.
    report["threshold_f1_optimal"] = report["threshold"]
    return report


@st.cache_data(ttl=60, show_spinner=False)
def sweep() -> list[dict]:
    return _get("/report/threshold-sweep")


@st.cache_data(ttl=60, show_spinner=False)
def cost_curves() -> dict:
    return _get("/report/cost-curve")


@st.cache_data(ttl=60, show_spinner=False)
def importance() -> list[dict]:
    return _get("/report/feature-importance")


@st.cache_data(ttl=60, show_spinner=False)
def segments() -> dict:
    return _get("/segments")


@st.cache_data(ttl=60, show_spinner=False)
def fairness() -> dict:
    """Just the per-attribute detail, which is what the fairness tab renders."""
    return _get("/fairness")["detail"]


@st.cache_data(ttl=60, show_spinner=False)
def fairness_verdict() -> dict:
    return _get("/fairness")


@st.cache_data(ttl=60, show_spinner=False)
def registry() -> list[dict]:
    """Registered versions, flattened into the row shape the registry tab draws."""
    rows = []
    for version in _get("/model-versions"):
        metrics = version.get("metrics") or {}
        rows.append(
            {
                "version": version["version"],
                "trained": version["trained_on"][:10],
                "roc_auc": metrics.get("roc_auc"),
                "f1": metrics.get("f1"),
                "fairness_passed": metrics.get("fairness_passed"),
                "fairness_total": metrics.get("fairness_total"),
                "status": "live" if version["is_live"] else "candidate",
            }
        )
    return rows


@st.cache_data(ttl=30, show_spinner=False)
def drift() -> dict:
    return _get("/monitoring/drift")


@st.cache_data(ttl=30, show_spinner=False)
def performance() -> dict:
    return _get("/monitoring/performance")


def explain(customer: dict) -> dict:
    """Score a hypothetical customer without recording it. Never cached."""
    return _post("/explain", customer)


# --- officer --------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def search_customers(q: str, limit: int = 25) -> list[dict]:
    """Find a customer by name, email or account number."""
    return _get("/officer/customers", q=q, limit=limit)


@st.cache_data(ttl=10, show_spinner=False)
def customer_detail(customer_id: str) -> dict:
    return _get(f"/officer/customers/{customer_id}")


def predict(customer_id: str, scored_by: str) -> dict:
    """The officer's predict action. Stores a prediction, so never cached."""
    return _post(f"/officer/customers/{customer_id}/predict", scored_by=scored_by)


def predict_selected(customer_ids: list[str], scored_by: str) -> list[dict]:
    """Predict for up to ten ticked customers. Stores each score, so never cached."""
    return _request(
        "/officer/predictions",
        payload={"customer_ids": customer_ids, "scored_by": scored_by},
        timeout=CHAT_TIMEOUT_SECONDS,
    )


@st.cache_data(ttl=300, show_spinner=False)
def briefing() -> dict:
    """Loana's opening: text, each quick cohort's label, filter and size, and a lead row.
    The backend stores one per day, so this is a short cache over a daily copy."""
    return _get("/officer/briefing")


def cohort(criteria: dict) -> dict:
    """The customers a triage filter selects: description, total, rows. Not cached:
    a row's latest band changes as soon as anyone predicts."""
    return _request("/officer/cohort", payload=criteria)


def triage(messages: list[dict], asked_by: str | None = None) -> dict:
    """Loana's triage turn: reply, source, note, filter and the cohort it selects."""
    return _request(
        "/officer/triage", payload={"messages": messages, "asked_by": asked_by},
        timeout=CHAT_TIMEOUT_SECONDS,
    )


@st.cache_data(ttl=5, show_spinner=False)
def review_requests() -> list[dict]:
    """Customers waiting for a human review: request_id, customer_id, account_number,
    full_name, requested_at, reason, status (pending or in_review)."""
    return _get("/officer/review-requests")


def officer_chat(customer_id: str, messages: list[dict]) -> dict:
    """The officer assistant's next turn. Never cached.

    Its own endpoint, not the cardholder's: the two are scoped to different readers.
    """
    return _request(
        f"/officer/customers/{customer_id}/chat",
        payload={"messages": messages},
        timeout=CHAT_TIMEOUT_SECONDS,
    )


def decide(
    customer_id: str, decided_by: str, action: str, justification: str,
    risk_band: str | None = None,
) -> dict:
    """Record a decision, setting the risk level when given; the backend also closes
    the customer's open conversation."""
    decision = _post(
        f"/officer/customers/{customer_id}/decisions",
        {
            "decided_by": decided_by, "action": action,
            "justification": justification, "risk_band": risk_band,
        },
    )
    _refresh_reviews()
    return decision


def approve_review(customer_id: str, decided_by: str) -> None:
    """Accept the customer's waiting requests so the conversation can open."""
    _post(f"/officer/customers/{customer_id}/review-requests/approve", {"decided_by": decided_by})
    _refresh_reviews()


def reject_review(customer_id: str, decided_by: str) -> None:
    """Decline the customer's open requests; no decision is issued."""
    _post(f"/officer/customers/{customer_id}/review-requests/reject", {"decided_by": decided_by})
    _refresh_reviews()


def _refresh_reviews() -> None:
    decisions.clear()
    thread.clear()
    review_requests.clear()
    customer_detail.clear()
    cardholder.clear()


@st.cache_data(ttl=5, show_spinner=False)
def thread(customer_id: str) -> list[dict]:
    """The account's messages and decisions, oldest first: id, kind (message or
    decision), at, author, body, action (decisions), status (messages)."""
    return _get(f"/officer/customers/{customer_id}/thread")


@st.cache_data(ttl=5, show_spinner=False)
def decisions(customer_id: str) -> list[dict]:
    return _get(f"/officer/customers/{customer_id}/decisions")


# --- cardholder -----------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def cardholder(customer_id: str, token: str) -> dict:
    """Deliberately narrower than the officer's view: no probability is sent."""
    # The token is an argument rather than read from session state here, so it is
    # part of the cache key: two cardholders must never share a response.
    return _request(f"/cardholders/{customer_id}", token=token)


def open_review_request(customer_id: str, token: str) -> dict | None:
    """This cardholder's open request for a human review, or None. Never cached."""
    return _request(f"/cardholders/{customer_id}/review-request", token=token)


def report_issue(token: str, kind: str, details: str) -> dict:
    """Store a problem or feedback from the signed-in person. Never cached."""
    return _request("/issue-reports", payload={"kind": kind, "details": details}, token=token)


def request_review(customer_id: str, token: str, reason: str) -> dict:
    """Send the loan officer a message; it opens (or reopens) the conversation."""
    sent = _request(
        f"/cardholders/{customer_id}/review-request",
        payload={"reason": reason},
        token=token,
    )
    review_requests.clear()
    thread.clear()
    return sent


def cardholder_thread(customer_id: str, token: str) -> list[dict]:
    """The same thread the officer reads. Never cached, so a reply shows on the next run."""
    return _request(f"/cardholders/{customer_id}/thread", token=token)


@st.cache_data(ttl=60, show_spinner=False)
def recourse(customer_id: str, token: str) -> dict:
    """The smallest change that would move this account to a better band.

    Cached longer than the account itself: the search runs the model dozens of
    times, and the answer only moves when the account's own figures do.
    """
    return _request(f"/cardholders/{customer_id}/recourse", token=token)


# OpenAI is cut off at 4 s by the API and the one-month search behind a turn takes
# about a second, so 20 s leaves room for both. A local Ollama model on CPU can
# take longer; the call then fails here with the usual message while the API goes
# on to log its own template fallback.
CHAT_TIMEOUT_SECONDS = 20


def chat(customer_id: str, token: str, messages: list[dict]) -> dict:
    """The assistant's next turn. Never cached: every turn is a new question."""
    return _request(
        f"/cardholders/{customer_id}/chat",
        payload={"messages": messages},
        token=token,
        timeout=CHAT_TIMEOUT_SECONDS,
    )


# --- derived --------------------------------------------------------------


def health() -> tuple[str, str, str]:
    """Status light: (colour, label, reason)."""
    verdict = fairness_verdict()
    if not verdict["passes"]:
        return (
            "🔴", "FAIRNESS BREACH",
            f"{', '.join(verdict['breaches'])} below the four-fifths rule",
        )

    auc = config()["test_metrics"]["roc_auc"]
    if abs(auc - BASELINE_AUC) > AUC_TOLERANCE:
        return "🟠", "DRIFT WARNING", f"test AUC {auc:.3f} against baseline {BASELINE_AUC:.3f}"

    return "🟢", "HEALTHY", "all fairness checks pass, AUC within tolerance"


def nearest_sweep_row(threshold: float) -> dict:
    """Closest swept row to a threshold, since the slider moves continuously."""
    return min(sweep(), key=lambda row: abs(row["threshold"] - threshold))
