"""Smoke checks for the officer's assistant.

Same guarantees as the cardholder chat: it always answers, the model restates the
fact sheet only, and nothing sent to the model carries the probability.

Run from the backend/ directory:

    python -m pytest tests/test_officer_assistant.py
"""
import json

import pytest
from app.ai import llm_client, officer_assistant
from app.ml import worklist
from tests.test_coaching import STRESSED


@pytest.fixture
def row():
    rows = worklist.all_rows()
    if not rows:
        pytest.skip("no queue artifact; run: python -m app.ml.train")
    return next(r for r in rows if r["risk_band"] == "High")


def _ask(question="Why is this customer flagged?"):
    return [{"role": "user", "content": question}]


def _llm(text=None, failure=None):
    return llm_client.LlmCall(
        text=text, model="test-model", latency_ms=12, failure=failure,
        prompt_tokens=100 if text else None, completion_tokens=20 if text else None,
    )


def test_without_the_model_a_template_summary_answers(row):
    reply = officer_assistant.answer(row, _ask(), record=STRESSED)

    assert reply.source == "template" and reply.note
    assert "The customer is rated High risk" in reply.reply
    assert "you" not in reply.reply.lower().split()


def test_a_grounded_model_reply_is_passed_through(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm("The customer is rated High risk."))

    reply = officer_assistant.answer(row, _ask(), record=STRESSED)

    assert (reply.source, reply.reply, reply.note) == (
        "model", "The customer is rated High risk.", ""
    )


def test_a_reply_with_an_invented_number_becomes_the_template(row, monkeypatch):
    monkeypatch.setattr(llm_client, "chat", lambda m: _llm("The risk is 87.31% for 99 months."))
    turns = []

    reply = officer_assistant.answer(row, _ask(), record=STRESSED, on_turn=turns.append)

    assert reply.source == "template"
    assert turns[0].fallback_reason == "invented_numbers"
    assert turns[0].rejected_reply.startswith("The risk is 87.31%")
    assert turns[0].prompt_version == officer_assistant.PROMPT_VERSION


def test_the_model_is_sent_no_probability(row, monkeypatch):
    sent = []
    monkeypatch.setattr(llm_client, "chat", lambda m: sent.append(m) or _llm(failure="llm_unreachable"))

    officer_assistant.answer(row, _ask(), record=STRESSED)

    prompt = json.dumps(sent[0])
    assert f"{row['probability']:.2f}" not in prompt
    assert "probability" not in prompt.split("FACTS:")[1].lower()
    assert sent[0][-1] == {"role": "user", "content": "Why is this customer flagged?"}
