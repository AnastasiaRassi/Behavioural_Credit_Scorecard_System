"""Plain-language coaching for a cardholder: what is raising the rating, and what one month could change.

Every number and action in the text comes from a `CoachingPayload` assembled in code.
The same payload is the counterfactual half of the assistant's fact sheet
(`assistant.py`), so the coaching card and the chat cannot disagree.

The text is a template, deliberately: it renders on every page load, and an LLM call is
too slow for that. The LLM lives in the chat (`assistant.py`), where a wait is expected;
this text is also the fallback for Loana's opening interpretation.

The probability is absent: `CardholderView` does not send it, so the text does not
quote it either.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from app.ml import artifacts
from app.ml.artifacts import ArtifactsMissing
from app.ml.counterfactuals import NEXT_BAND_UP, CounterfactualExplainer, RecourseTarget
from app.ml.explainability import describe_change

MAX_REASONS = 3
DISCLAIMER = (
    "This is a model's estimate on a prototype. It is not a credit decision, "
    "a guarantee, or financial advice."
)

LOWEST_BAND = "lowest_band"  # already as good as the model rates anyone
REACHABLE = "reachable"
UNREACHABLE = "unreachable"  # the search found no single-month action that clears it
UNCHECKED = "unchecked"  # no raw account record, so the search could not run

# Written by db/seed.py for the demo cardholders. train.py keeps it (PRESERVES).
CACHE_NAME = "coaching.json"

# The page and every chat turn need the same payload; the search behind it takes
# about a second, so recent ones are kept.
PAYLOAD_MEMO_SIZE = 256

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_payload_memo: OrderedDict[str, CoachingPayload] = OrderedDict()


@dataclass(frozen=True)
class CoachingPayload:
    """Every fact the text may state. Built in code before any wording happens."""

    band: str
    reasons: tuple[str, ...]
    target_band: str | None
    outcome: str
    steps: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Coaching:
    text: str
    disclaimer: str

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _explainer() -> CounterfactualExplainer:
    return CounterfactualExplainer()


def raises_risk(reason: Mapping) -> bool:
    """Stored reasons carry the direction; queue rows carry the effect it came from."""
    return reason.get("raises_risk", reason.get("effect", 0) > 0)


def build_payload(
    band: str, reasons: Sequence[Mapping], record: Mapping[str, float] | None = None
) -> CoachingPayload:
    """The facts for one account. `record` is the raw input, needed for the one-month search."""
    key = json.dumps(
        [band, [reason["description"] for reason in reasons], record], sort_keys=True, default=str
    )
    if key in _payload_memo:
        _payload_memo.move_to_end(key)
        return _payload_memo[key]

    payload = _build(band, reasons, record)
    _payload_memo[key] = payload
    if len(_payload_memo) > PAYLOAD_MEMO_SIZE:
        _payload_memo.popitem(last=False)
    return payload


def _build(band: str, reasons: Sequence[Mapping], record: Mapping[str, float] | None) -> CoachingPayload:
    raising = tuple(reason["description"] for reason in reasons if raises_risk(reason))[:MAX_REASONS]

    target = NEXT_BAND_UP.get(band)
    if target is None:
        return CoachingPayload(band, raising, None, LOWEST_BAND, ())
    if record is None:
        return CoachingPayload(band, raising, target, UNCHECKED, ())

    result = _explainer().explain(dict(record), RecourseTarget(band=target))
    if not result.is_feasible:
        return CoachingPayload(band, raising, target, UNREACHABLE, ())

    steps = tuple(
        describe_change(feature, change["from"], change["to"])
        for feature, change in result.feature_changes.items()
    )
    return CoachingPayload(band, raising, target, REACHABLE, steps)


def render(payload: CoachingPayload) -> str:
    """Plain sentences built only from the payload."""
    sentences = [f"On this model your account is rated {payload.band} risk for next month."]
    if payload.reasons:
        sentences.append("What is raising the rating: " + "; ".join(payload.reasons) + ".")
    else:
        sentences.append("Nothing in the recent account history is raising the rating.")
    sentences.append(action_sentence(payload))
    return " ".join(sentences)


def action_sentence(payload: CoachingPayload) -> str:
    """What one month could change, or why it cannot."""
    target = payload.target_band
    if payload.outcome == LOWEST_BAND:
        return "It is already in the lowest risk band, so there is no change to make."
    if payload.outcome == UNCHECKED:
        return (
            f"The one-month changes that would move it to {target} risk could not "
            "be checked just now."
        )
    if payload.outcome == UNREACHABLE:
        return (
            "No single-month action clears it: nothing this month's bill, payment or "
            f"repayment status can change would move it to {target} risk on this model."
        )
    if payload.steps:
        return (
            f"Doing all of these this month would move it to {target} risk on this "
            "model: " + "; ".join(payload.steps) + "."
        )
    return f"No change is needed to reach {target} risk on this model."


def numbers_in(text: str) -> set[str]:
    """Every number in `text`, thousands separators removed, so 15,000 matches 15000."""
    return {match.group().replace(",", "") for match in _NUMBER.finditer(text)}


def is_grounded(text: str, payload: CoachingPayload) -> bool:
    """True when every number in `text` is one the payload already contains."""
    return numbers_in(text) <= numbers_in(json.dumps(payload.to_dict()))


def coach(
    customer_id: str,
    band: str,
    reasons: Sequence[Mapping],
    record: Mapping[str, float] | None = None,
) -> Coaching:
    """Coaching for one account. Needs no network."""
    descriptions = [reason["description"] for reason in reasons]
    cached = _cached(customer_id, band, descriptions)
    if cached is not None:
        return cached
    return Coaching(text=render(build_payload(band, reasons, record)), disclaimer=DISCLAIMER)


def _model_version() -> str:
    return artifacts.scoring_config()["model_version"]


def _cached(customer_id: str, band: str, descriptions: list[str]) -> Coaching | None:
    """A pre-generated text, only while it still describes this model and this account."""
    try:
        cache = artifacts.coaching_cache()
    except ArtifactsMissing:
        return None
    if cache.get("model_version") != _model_version():
        return None
    entry = cache.get("entries", {}).get(customer_id)
    if entry is None or entry["band"] != band or entry["reasons"] != descriptions:
        return None
    return Coaching(text=entry["coaching"]["text"], disclaimer=entry["coaching"]["disclaimer"])


def cache_entry(
    customer_id: str,
    band: str,
    reasons: Sequence[Mapping],
    record: Mapping[str, float] | None,
) -> dict:
    """One account's cache record."""
    return {
        "band": band,
        "reasons": [reason["description"] for reason in reasons],
        "coaching": coach(customer_id, band, reasons, record).to_dict(),
    }


def cache_covers(customer_ids: Sequence[str]) -> bool:
    """True when the cache was written by the live model and holds every id asked for."""
    try:
        cache = artifacts.coaching_cache()
    except ArtifactsMissing:
        return False
    return cache.get("model_version") == _model_version() and set(customer_ids) <= set(
        cache.get("entries", {})
    )


def write_cache(entries: Mapping[str, dict]) -> None:
    document = {"model_version": _model_version(), "entries": dict(entries)}
    path = artifacts.artifacts_dir() / CACHE_NAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
