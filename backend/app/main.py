"""FastAPI application.

Run from the backend/ directory:

    uvicorn app.main:app --reload

Routes translate HTTP and nothing else: scoring lives in `ml/scorer.py`,
explanations in `ml/explainability.py`, drift in `ml/drift.py`, and every
reported metric is read from artifacts/ rather than recomputed.

`POST /session` verifies a password and returns a signed, expiring token; the
caller presents it as `Authorization: Bearer <token>` and the role is read from
the account, never from the request. Reading a cardholder's account is guarded
by it (`dependencies.account_reader`). The officer's own write paths still name
the acting officer by email in the request body, which records who acted without
proving it — the next thing to close.
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api import (
    admin,
    cardholders,
    decisions,
    fairness,
    monitoring,
    officer,
    predictions,
    recourse,
    roles,
    sessions,
)
from app.core.exceptions import AppError
from app.core.logging_config import configure_logging, request_id
from app.db.database import create_all
from app.ml import artifacts
from app.ml.exceptions import ArtifactsMissing

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Creating missing tables on startup keeps a prototype runnable from a clean
    # database. A migration tool would replace this if the schema had to evolve
    # against data worth keeping.
    create_all()
    try:
        config = artifacts.scoring_config()
        logger.info(
            "startup: model=%s trained=%s threshold=%s",
            config["model_version"],
            config["trained"],
            config.get("threshold_f1_optimal"),
        )
    except ArtifactsMissing as exception:
        logger.error("startup: artifacts missing, scoring will return 503: %s", exception)
    yield
    logger.info("shutdown")


app = FastAPI(
    lifespan=lifespan,
    title="Credit Risk · Behavioural Scorecard",
    description=(
        "Scores existing cardholders for the risk of missing their next payment, "
        "explains each score, and audits itself for fairness."
    ),
    version="1.0.0",
)

def _error_body(code: str, message: str) -> dict:
    # `detail` stays for callers written against FastAPI's default shape.
    return {"code": code, "message": message, "detail": message, "request_id": request_id.get()}


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exception: AppError) -> JSONResponse:
    """One log line with the code and context; the traceback only for our own faults."""
    if exception.status >= 500:
        logger.error(
            "%s %s -> %s", request.method, request.url.path, exception.log_line(),
            exc_info=exception,
        )
    else:
        logger.warning("%s %s -> %s", request.method, request.url.path, exception.log_line())
    return JSONResponse(
        status_code=exception.status,
        content=_error_body(exception.code, exception.user_message),
        headers={"X-Request-ID": request_id.get()},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exception: HTTPException) -> JSONResponse:
    message = exception.detail if isinstance(exception.detail, str) else "Request failed."
    headers = {**(exception.headers or {}), "X-Request-ID": request_id.get()}
    return JSONResponse(
        status_code=exception.status_code,
        content=_error_body(f"HTTP_{exception.status_code}", message),
        headers=headers,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request: method, path, status, time. The query string is left out
    because it carries emails and search terms.

    Anything uncaught below ends here: logged with its traceback and answered with a
    reference the caller can quote, instead of a bare 500."""
    token = request_id.set(uuid.uuid4().hex[:8])
    started = time.perf_counter()
    client = request.client.host if request.client else "-"
    try:
        try:
            response = await call_next(request)
        except Exception as exception:
            logger.exception(
                "%s %s -> INTERNAL_ERROR %s client=%s",
                request.method, request.url.path, type(exception).__name__, client,
            )
            response = JSONResponse(
                status_code=500,
                content=_error_body("INTERNAL_ERROR", "Something went wrong on our side."),
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        # Health checks poll constantly; keep them out of the INFO stream.
        level = logging.DEBUG if request.url.path == "/health" else (
            logging.WARNING if response.status_code >= 400 else logging.INFO
        )
        logger.log(
            level,
            "%s %s -> %s %dms client=%s",
            request.method, request.url.path, response.status_code, elapsed_ms, client,
        )
        response.headers["X-Request-ID"] = request_id.get()
        return response
    finally:
        request_id.reset(token)


app.include_router(sessions.router)
app.include_router(roles.router)
app.include_router(officer.router)
app.include_router(cardholders.router)
app.include_router(recourse.router)
app.include_router(predictions.router)
app.include_router(decisions.router)
app.include_router(fairness.router)
app.include_router(admin.router)
app.include_router(monitoring.router)


@app.get("/health", tags=["service"])
def health() -> dict:
    """Whether the service can actually serve: artifacts present, model live."""
    try:
        config = artifacts.scoring_config()
    except ArtifactsMissing as exception:
        return {"status": "unprovisioned", "detail": str(exception)}

    return {
        "status": "ok",
        "model_version": config["model_version"],
        "trained": config["trained"],
        "n_features": config["n_features"],
    }
