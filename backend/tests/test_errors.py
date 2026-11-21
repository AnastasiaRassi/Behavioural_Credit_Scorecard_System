"""The error contract: codes, safe messages, and what the API says when something breaks."""
import json
import logging

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DataError, OperationalError

from app.ai.exceptions import LlmTimeout, LlmUnreachable, error_for
from app.core.exceptions import AppError, ConfigError, InfraError, NotFoundError
from app.db.database import translate
from app.db.exceptions import (
    CustomerNotFound,
    DatabaseUnavailable,
    PersistenceFailed,
    SourceUnavailable,
)
from app.main import app
from app.ml.exceptions import ArtifactsMissing, FeatureSchemaMismatch, MissingColumns


def test_user_message_is_separate_from_the_technical_one():
    error = ArtifactsMissing("Missing x.joblib in /srv/models", context={"artifact": "x"})
    assert "/srv" in str(error)
    assert "/srv" not in error.user_message
    assert error.status == 503 and error.code == "ARTIFACTS_MISSING"
    assert error.log_line() == "ARTIFACTS_MISSING: Missing x.joblib in /srv/models artifact=x"


def test_layer_errors_keep_their_builtin_bases():
    assert issubclass(CustomerNotFound, LookupError)
    assert issubclass(SourceUnavailable, FileNotFoundError)
    assert issubclass(MissingColumns, ValueError)
    assert issubclass(FeatureSchemaMismatch, ValueError)
    assert issubclass(ArtifactsMissing, RuntimeError)
    assert issubclass(CustomerNotFound, NotFoundError)
    assert issubclass(SourceUnavailable, InfraError)


def test_there_is_one_artifacts_missing():
    from app.ml import artifacts, scorer, segmentation

    assert artifacts.ArtifactsMissing is scorer.ArtifactsMissing is segmentation.ArtifactsMissing


def test_config_errors_are_app_errors():
    from app.core.config import ConfigurationMissing

    assert issubclass(ConfigurationMissing, ConfigError) and issubclass(ConfigError, AppError)


def test_llm_failure_codes_map_to_typed_errors_without_content():
    assert isinstance(error_for("llm_timeout", "m", 4000), LlmTimeout)
    assert isinstance(error_for("llm_unreachable"), LlmUnreachable)
    assert error_for("llm_timeout", "m", 4000).context == {
        "failure": "llm_timeout", "model": "m", "latency_ms": 4000,
    }


def test_driver_errors_are_translated_without_parameters():
    data_error = DataError(
        "INSERT INTO prediction (explanation) VALUES (%(e)s)", {"e": "secret-value"},
        Exception("invalid input syntax for type json"),
    )
    translated = translate(data_error)
    assert isinstance(translated, PersistenceFailed)
    assert translated.context == {"table": "prediction"}
    assert "secret-value" not in translated.log_line()

    down = OperationalError("select 1", {}, Exception("connection refused"))
    assert isinstance(translate(down), DatabaseUnavailable)


def test_a_typed_error_is_answered_with_code_and_request_id():
    @app.get("/_boom_typed")
    def boom():
        raise ArtifactsMissing("Missing model.joblib in /secret/path")

    response = TestClient(app).get("/_boom_typed")
    body = response.json()
    assert response.status_code == 503
    assert body["code"] == "ARTIFACTS_MISSING"
    assert body["request_id"] == response.headers["X-Request-ID"] != "-"
    assert "/secret/path" not in json.dumps(body)


def test_an_uncaught_error_is_logged_with_its_traceback_and_a_reference(caplog):
    @app.get("/_boom_untyped")
    def boom():
        raise RuntimeError("kaboom")

    with caplog.at_level(logging.ERROR):
        response = TestClient(app, raise_server_exceptions=False).get("/_boom_untyped")
    body = response.json()
    assert response.status_code == 500
    assert body["code"] == "INTERNAL_ERROR"
    assert "kaboom" not in json.dumps(body)
    record = next(r for r in caplog.records if "INTERNAL_ERROR" in r.getMessage())
    assert record.exc_info and "kaboom" in str(record.exc_info[1])


def test_a_caller_error_shows_its_message_and_keeps_its_headers():
    response = TestClient(app).get("/session")
    body = response.json()
    assert response.status_code == 401
    assert body["code"] == "AUTH_REQUIRED" and body["detail"] == body["message"]
    assert "Bearer" in body["message"]
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_plain_http_error_still_gains_a_code():
    response = TestClient(app).get("/no-such-route")
    assert response.status_code == 404
    assert response.json()["code"] == "HTTP_404"


def test_no_route_raises_a_bare_http_exception():
    import pathlib

    api = pathlib.Path(__file__).resolve().parents[1] / "app" / "api"
    offenders = [p.name for p in api.glob("*.py") if "raise HTTPException" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_remaining_value_errors_are_typed_and_still_value_errors():
    from app.core.passwords import hash_password
    from app.ml.exceptions import UnknownBand

    with pytest.raises(ValueError) as raised:
        hash_password("")
    assert raised.value.code == "EMPTY_PASSWORD"
    assert issubclass(UnknownBand, ValueError)


def test_a_failed_chat_turn_write_is_logged_with_a_code(caplog):
    from app.db import chat_log

    with caplog.at_level(logging.ERROR):
        chat_log.log_failure("chat log:", DataError("INSERT INTO chat_turn (x) VALUES (%(x)s)", {"x": "private"}, Exception("bad json")))
    line = caplog.records[-1].getMessage()
    assert "PERSISTENCE_FAILED" in line and "table=chat_turn" in line and "private" not in line
