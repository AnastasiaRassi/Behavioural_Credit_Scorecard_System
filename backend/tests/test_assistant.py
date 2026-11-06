"""Smoke checks for the cardholder chat.

The contract: the assistant always answers, the local model only restates the fact
sheet, and a model failure or a reply with an invented number becomes a template
answer. No Ollama is needed to run these; the model is replaced per test.

Run from the backend/ directory:

    python -m pytest tests/test_assistant.py
"""
import json

import pytest
import requests

from app.ai import assistant, coaching, local_llm
from app.ml import worklist
from tests.test_coaching import STRESSED


@pytest.fixture
def row():
    rows = worklist.all_rows()
    if not rows:
        pytest.skip("no queue artifact; run: python -m app.ml.train")
    return next(r for r in rows if r["risk_band"] == "High")


def _ask(question):
    return [{"role": "user", "content": question}]


def _sheet(row):
    view = worklist.cardholder_view(row, record=STRESSED)
    payload = coaching.build_payload(view["risk_band"], view["reasons"], STRESSED)
    return assistant.fact_sheet(view, row["history"], payload)


def test_empty_conversation_returns_the_greeting(row):
    reply = assistant.answer(row, [], record=STRESSED)

    assert reply.source == "template"
    assert row["risk_band"] in reply.reply


def test_answers_without_the_model_and_every_number_is_grounded(row, monkeypatch):
    monkeypatch.setattr(local_llm, "chat", lambda messages: None)

    for question in ("Why am I high risk?", "What can I do?", "What is my balance?",
                     "What is my probability?", "Hello"):
        reply = assistant.answer(row, _ask(question), record=STRESSED)
        assert reply.source == "template"
        assert reply.reply.strip()
        assert reply.note
        assert coaching.numbers_in(reply.reply) <= coaching.numbers_in(json.dumps(_sheet(row)))


def test_fact_sheet_carries_counterfactual_reasons_and_history_but_no_probability(row):
    sheet = _sheet(row)

    assert sheet["one_month_counterfactual"]["target_band"] == "Medium"
    assert sheet["factors_raising_risk"]
    assert len(sheet["recent_months"]) == len(row["history"])
    assert str(row["probability"]) not in json.dumps(sheet)
    assert "probability" not in json.dumps(sheet)


def test_model_reply_with_an_invented_number_is_replaced(row, monkeypatch):
    monkeypatch.setattr(local_llm, "chat", lambda messages: "Pay 987,654 and you will be Low risk.")

    reply = assistant.answer(row, _ask("What can I do?"), record=STRESSED)

    assert reply.source == "template"
    assert "987" not in reply.reply


def test_grounded_model_reply_is_used(row, monkeypatch):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return f"Your account is rated {row['risk_band']} risk on this model."

    monkeypatch.setattr(local_llm, "chat", fake_chat)

    reply = assistant.answer(row, _ask("Why?"), record=STRESSED)

    assert reply.source == "model"
    assert reply.note == ""
    assert seen["messages"][0]["role"] == "system"
    assert "one_month_counterfactual" in seen["messages"][0]["content"]


def test_numbers_the_cardholder_typed_may_be_echoed(row, monkeypatch):
    monkeypatch.setattr(local_llm, "chat", lambda messages: "I cannot estimate what paying 4,321 would do.")

    reply = assistant.answer(row, _ask("What if I pay 4321?"), record=STRESSED)

    assert reply.source == "model"


def test_local_llm_returns_none_when_ollama_is_down(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", refuse)
    assert local_llm.chat([{"role": "user", "content": "hi"}]) is None


def test_local_llm_returns_none_on_timeout(monkeypatch):
    def slow(*args, **kwargs):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(requests, "post", slow)
    assert local_llm.chat([{"role": "user", "content": "hi"}]) is None
