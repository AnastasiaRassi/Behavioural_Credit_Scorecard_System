"""Counterfactual recourse: the smallest change that moves a customer into a better band.

SHAP and friends say *why* the score is what it is. This says *what to do about it*,
which is the question a declined customer actually asks.

Perturbations are applied to the raw columns, never to the engineered features. The
engineered set is derived (`util_*`, `repay_ratio_*`, `remaining_credit`), so editing
it directly produces a feature vector no real customer could have, and recourse the
bank cannot honour. Changing `BILL_AMT1` and re-running `engineer()` keeps every
derived value consistent.

Only the current month is actionable: pay the card down (`BILL_AMT1`), pay more into it
(`PAY_AMT1`), bring the account current (`PAY_1`). History is history: `PAY_2..PAY_6`,
the older bills and the older payments are facts, and protected attributes are excluded
from the model anyway. `LIMIT_BAL` is the lender's lever rather than the customer's, so
it is off by default and can be enabled when modelling a limit increase.

Clearing arrears is coupled to paying for it. `ArrearsConsistency` enforces that, since
an unconstrained search will happily improve `PAY_1` while leaving the payment at zero —
a better score, and advice nobody can act on.

Two searches sit behind one interface. `GreedySearch` walks the actionable features a
step at a time and is the default; `RuleBasedSearch` evaluates a handful of
closed-form candidates ("settle the bill", "clear the balance") in a single batch and
runs in milliseconds. The explainer falls back to the second whenever the first times
out, exhausts its budget or raises, so a caller always gets an answer.

Run `python -m app.ml.counterfactuals` for a demonstration on one customer.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from app.ml.feature_engineering import SCORED_COLUMNS
from app.ml.scorer import CreditScorer

# Search budget. Generous enough to find recourse on hard cases, bounded so an API
# request cannot hang on one.
MAX_ITERATIONS = 60
TIMEOUT_SECONDS = 5.0

# Step multipliers tried per feature per iteration: a fine move for precision, coarse
# ones so a customer far from the threshold is not walked there 1% at a time.
STEP_MULTIPLIERS = (1.0, 3.0, 10.0)

# A change smaller than this is rounding, not advice.
NEGLIGIBLE_DELTA = 1e-6

# PAY_n == 0 is revolving credit, the best status reachable by paying: -1 ("paid in
# full") and -2 ("no balance") describe a customer who had nothing to settle.
CURRENT_STATUS = 0.0

# Fractions of each lever's available range that the rule-based search evaluates.
RULE_STRENGTHS = (0.25, 0.5, 0.75, 1.0)

# Share of the outstanding bill assumed to be the minimum due per month of arrears.
MIN_PAYMENT_SHARE = 0.10

# One band better is the target worth offering. Asking a High-risk account to reach
# Low in a month is advice nobody can act on, and the search would spend its budget
# proving it.
NEXT_BAND_UP = {"High": "Medium", "Medium": "Low"}


class Direction:
    """Allowed sign of movement for an actionable feature."""

    UP = "increase"
    DOWN = "decrease"
    ANY = "any"


@dataclass(frozen=True)
class ActionBound:
    """Range, granularity and allowed direction for one actionable feature."""

    minimum: float
    maximum: float
    step: float
    direction: str = Direction.ANY
    cost_weight: float = 1.0
    integral: bool = False  # status codes have no meaningful fractional value

    def clip(self, value: float) -> float:
        bounded = min(max(value, self.minimum), self.maximum)
        return float(round(bounded) if self.integral else bounded)

    def span(self) -> float:
        return max(self.maximum - self.minimum, NEGLIGIBLE_DELTA)


@dataclass(frozen=True)
class RecoursePolicy:
    """Which features may move, how far, and what counts as expensive.

    `bounds` are relative to the customer being explained, so they are built per call
    by `default_policy`. Anything not named here is immutable by construction.
    """

    bounds: Mapping[str, ActionBound]

    def actionable(self) -> tuple[str, ...]:
        return tuple(self.bounds)

    def without(self, features: Iterable[str]) -> RecoursePolicy:
        excluded = set(features)
        return RecoursePolicy({k: v for k, v in self.bounds.items() if k not in excluded})

    def cost(self, original: Mapping[str, float], candidate: Mapping[str, float]) -> float:
        """Weighted L1 distance, normalised per feature so units do not dominate."""
        return sum(
            bound.cost_weight
            * abs(float(candidate[name]) - float(original[name]))
            / bound.span()
            for name, bound in self.bounds.items()
        )


@dataclass(frozen=True)
class RecourseTarget:
    """Where the customer needs to get to, as a band or an explicit probability."""

    band: str | None = "Low"
    max_probability: float | None = None

    def resolve(self, scorer: CreditScorer) -> float:
        """Highest probability that still satisfies this target."""
        if self.max_probability is not None:
            return float(self.max_probability)
        if self.band is None:
            return float(scorer.threshold)
        edges = {"Low": scorer.band_edges["low_max"], "Medium": scorer.band_edges["medium_max"]}
        if self.band not in edges:
            raise ValueError(f"No ceiling for band {self.band!r}; expected one of {sorted(edges)}")
        # Band membership is strict (`probability < edge`), so step just inside it.
        return float(edges[self.band]) - 1e-4


@dataclass
class Counterfactual:
    """The recourse offered to one customer, or the evidence that none was found."""

    original_score: float
    target_score: float
    achieved_score: float
    feature_changes: dict[str, dict[str, float]]
    is_feasible: bool
    original_band: str = ""
    achieved_band: str = ""
    method: str = ""
    iterations: int = 0
    elapsed_seconds: float = 0.0
    total_cost: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ProbabilityModel(Protocol):
    """Anything that can score raw customer records. Decouples search from the model."""

    def probabilities(self, customers: Sequence[Mapping[str, float]]) -> np.ndarray: ...

    def band(self, probability: float) -> str: ...


class ScorerModel:
    """Adapts `CreditScorer` to `ProbabilityModel`, scoring candidates in one batch."""

    def __init__(self, scorer: CreditScorer):
        self.scorer = scorer

    def probabilities(self, customers: Sequence[Mapping[str, float]]) -> np.ndarray:
        return np.asarray(self.scorer.probabilities(pd.DataFrame(list(customers))), dtype=float)

    def band(self, probability: float) -> str:
        return self.scorer.band(probability)


class CounterfactualSearch(Protocol):
    """A strategy for finding a feasible, low-cost perturbation."""

    name: str

    def search(
        self,
        customer: Mapping[str, float],
        policy: RecoursePolicy,
        target: float,
        model: ProbabilityModel,
        deadline: float,
    ) -> SearchOutcome: ...


@dataclass
class SearchOutcome:
    """Raw result of one strategy, before it is described to the caller."""

    candidate: dict[str, float]
    probability: float
    feasible: bool
    iterations: int
    note: str = ""


# ---- policy ----------------------------------------------------------------


def default_policy(
    customer: Mapping[str, float],
    *,
    allow_clear_arrears: bool = True,
    allow_limit_increase: bool = False,
    settle_multiple: float = 1.0,
) -> RecoursePolicy:
    """Actionable bounds for this customer: pay down the bill, pay more, clear arrears.

    `settle_multiple` caps the suggested payment at that multiple of the bill being
    repaid, so recourse stays inside what the customer could plausibly transfer.
    """
    limit = float(customer["LIMIT_BAL"])
    bill_now = float(customer["BILL_AMT1"])
    prior_bill = max(float(customer["BILL_AMT2"]), 0.0)
    paid_now = float(customer["PAY_AMT1"])

    payment_ceiling = max(prior_bill * settle_multiple, paid_now, limit * 0.05)
    bounds = {
        # Paying the card down is the one lever that is unambiguously the customer's.
        "BILL_AMT1": ActionBound(
            minimum=0.0,
            maximum=max(bill_now, 0.0),
            step=max(abs(bill_now), limit) * 0.05,
            direction=Direction.DOWN,
        ),
        "PAY_AMT1": ActionBound(
            minimum=paid_now,
            maximum=payment_ceiling,
            step=max(payment_ceiling - paid_now, limit * 0.01) * 0.05,
            direction=Direction.UP,
        ),
    }
    status_now = float(customer["PAY_1"])
    if allow_clear_arrears and status_now > CURRENT_STATUS:
        # Bringing the account current this month. The lever the score responds to most,
        # and only credible alongside the payment `ArrearsConsistency` enforces.
        bounds["PAY_1"] = ActionBound(
            minimum=CURRENT_STATUS,
            maximum=status_now,
            step=1.0,
            direction=Direction.DOWN,
            integral=True,
        )
    if allow_limit_increase:
        # A lender-side action, and an expensive one: weighted up so the search
        # prefers customer behaviour when either would work.
        bounds["LIMIT_BAL"] = ActionBound(
            minimum=limit,
            maximum=limit * 2.0,
            step=limit * 0.05,
            direction=Direction.UP,
            cost_weight=3.0,
        )
    return RecoursePolicy(bounds)


class ArrearsConsistency:
    """Forces a counterfactual to pay for the repayment status it claims.

    Without this a search is free to set `PAY_1` to "current" while leaving
    `PAY_AMT1` at zero: a strictly better score, and advice no customer can act on.
    The minimum due is taken as `MIN_PAYMENT_SHARE` of the prior bill per month of
    arrears cleared, with full settlement required to reach current.
    """

    def repair(self, original: Mapping[str, float], candidate: Mapping[str, float]) -> dict:
        """Raise the current month's payment to cover the arrears the candidate clears."""
        fixed = dict(candidate)
        before, after = float(original["PAY_1"]), float(fixed.get("PAY_1", original["PAY_1"]))
        cleared = before - after
        if cleared <= 0:
            return fixed

        prior_bill = max(float(original["BILL_AMT2"]), 0.0)
        due = prior_bill if after <= CURRENT_STATUS else prior_bill * MIN_PAYMENT_SHARE * cleared
        fixed["PAY_AMT1"] = max(float(fixed["PAY_AMT1"]), due)
        return fixed


class NoConsistency:
    """Opt-out for callers who want the raw nearest counterfactual."""

    def repair(self, original: Mapping[str, float], candidate: Mapping[str, float]) -> dict:
        return dict(candidate)


class ConsistentModel:
    """Decorates a model so every candidate is repaired before it is scored."""

    def __init__(self, model: ProbabilityModel, consistency, original: Mapping[str, float]):
        self.model = model
        self.consistency = consistency
        self.original = original

    def probabilities(self, customers: Sequence[Mapping[str, float]]) -> np.ndarray:
        return self.model.probabilities(
            [self.consistency.repair(self.original, c) for c in customers]
        )

    def band(self, probability: float) -> str:
        return self.model.band(probability)


# ---- search strategies -----------------------------------------------------


def _moves(value: float, bound: ActionBound) -> list[float]:
    """Candidate next values for one feature, respecting direction and bounds."""
    signs = {Direction.UP: (1,), Direction.DOWN: (-1,), Direction.ANY: (1, -1)}[bound.direction]
    candidates = {
        bound.clip(value + sign * bound.step * multiplier)
        for sign in signs
        for multiplier in STEP_MULTIPLIERS
    }
    return [c for c in candidates if abs(c - value) > NEGLIGIBLE_DELTA]


class GreedySearch:
    """Coordinate descent on the actionable features, best score drop per unit cost wins.

    Deliberately not gradient-based. The model is a LightGBM ensemble behind a
    calibrator, so there is no useful gradient to follow, and a step this size is
    cheap when every candidate is scored in one batch.
    """

    name = "greedy-coordinate"

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.max_iterations = max_iterations

    def search(
        self,
        customer: Mapping[str, float],
        policy: RecoursePolicy,
        target: float,
        model: ProbabilityModel,
        deadline: float,
    ) -> SearchOutcome:
        state = dict(customer)
        probability = float(model.probabilities([state])[0])
        iterations = 0

        while probability > target and iterations < self.max_iterations:
            if time.perf_counter() >= deadline:
                return SearchOutcome(state, probability, False, iterations, "timed out")

            moves = [
                (name, value)
                for name, bound in policy.bounds.items()
                for value in _moves(float(state[name]), bound)
            ]
            if not moves:
                return SearchOutcome(state, probability, False, iterations, "bounds exhausted")

            scores = model.probabilities([{**state, name: value} for name, value in moves])
            best = self._best_move(state, policy, moves, scores, probability)
            iterations += 1
            if best is None:
                return SearchOutcome(state, probability, False, iterations, "no improving move")

            (name, value), probability = best
            state[name] = value

        feasible = probability <= target
        return SearchOutcome(
            state, probability, feasible, iterations, "" if feasible else "iteration budget spent"
        )

    @staticmethod
    def _best_move(
        state: Mapping[str, float],
        policy: RecoursePolicy,
        moves: Sequence[tuple[str, float]],
        scores: np.ndarray,
        current: float,
    ) -> tuple[tuple[str, float], float] | None:
        """Move with the largest score drop per unit of added cost, or None if none helps."""
        best: tuple[tuple[str, float], float] | None = None
        best_ratio = 0.0
        for (name, value), score in zip(moves, scores):
            gain = current - float(score)
            if gain <= NEGLIGIBLE_DELTA:
                continue
            added = policy.cost(state, {**state, name: value})
            ratio = gain / max(added, NEGLIGIBLE_DELTA)
            if ratio > best_ratio:
                best_ratio, best = ratio, ((name, value), float(score))
        return best


class RuleBasedSearch:
    """Score a few domain-shaped candidates in one batch, then tighten the winner.

    The fallback. It cannot explore combinations the greedy walk would find, but it is
    one or two model calls, never loops on a flat score surface, and the candidates are
    the actions a credit officer would propose anyway.
    """

    name = "rule-based"

    def __init__(self, refine_steps: int = 12):
        self.refine_steps = refine_steps

    def search(
        self,
        customer: Mapping[str, float],
        policy: RecoursePolicy,
        target: float,
        model: ProbabilityModel,
        deadline: float,
    ) -> SearchOutcome:
        base = dict(customer)
        candidates = self._candidates(base, policy)
        if not candidates:
            probability = float(model.probabilities([base])[0])
            return SearchOutcome(base, probability, probability <= target, 0, "no rules apply")

        scores = model.probabilities([base, *candidates])
        current, candidate_scores = float(scores[0]), scores[1:]
        if current <= target:
            return SearchOutcome(base, current, True, 1, "")

        feasible = [
            (policy.cost(base, candidate), candidate, float(score))
            for candidate, score in zip(candidates, candidate_scores)
            if float(score) <= target
        ]
        if not feasible:
            best_index = int(np.argmin(candidate_scores))
            return SearchOutcome(
                candidates[best_index],
                float(candidate_scores[best_index]),
                False,
                1,
                "no rule reaches the target",
            )

        _, winner, score = min(feasible, key=lambda item: item[0])
        if time.perf_counter() < deadline:
            winner, score = self._refine(base, winner, policy, target, model)
        return SearchOutcome(winner, score, True, 1 + self.refine_steps, "")

    @staticmethod
    def _candidates(
        customer: Mapping[str, float], policy: RecoursePolicy
    ) -> list[dict[str, float]]:
        """Each lever at a few strengths, alone and in combination.

        Not just the extremes: the ensemble is not monotonic in the bill, so "clear the
        card entirely" can score worse than paying most of it down.
        """
        candidates: list[dict[str, float]] = []
        for strength in RULE_STRENGTHS:
            combined = dict(customer)
            for name, bound in policy.bounds.items():
                current = float(customer[name])
                extreme = bound.minimum if bound.direction == Direction.DOWN else bound.maximum
                moved = bound.clip(current + (extreme - current) * strength)
                if abs(moved - current) <= NEGLIGIBLE_DELTA:
                    continue
                candidates.append({**customer, name: moved})
                combined[name] = moved
            if len(policy.bounds) > 1 and combined != dict(customer):
                candidates.append(combined)
        return candidates

    def _refine(
        self,
        original: Mapping[str, float],
        winner: Mapping[str, float],
        policy: RecoursePolicy,
        target: float,
        model: ProbabilityModel,
    ) -> tuple[dict[str, float], float]:
        """Shrink the winning candidate to the smallest version that still works."""
        return tighten(original, winner, policy, target, model, self.refine_steps)


def tighten(
    original: Mapping[str, float],
    candidate: Mapping[str, float],
    policy: RecoursePolicy,
    target: float,
    model: ProbabilityModel,
    steps: int = 12,
) -> tuple[dict[str, float], float]:
    """Bisect each changed feature back toward its original value while still feasible.

    Both strategies overshoot: the greedy walk takes coarse steps and the rules jump
    straight to the extreme. This is what makes the reported recourse near-minimal.
    """
    refined = dict(candidate)
    score = float(model.probabilities([refined])[0])

    for name in policy.actionable():
        start, end = float(original[name]), float(refined[name])
        if abs(end - start) <= NEGLIGIBLE_DELTA:
            continue
        bound = policy.bounds[name]
        low, high = start, end  # low may be infeasible, high is known feasible
        for _ in range(steps):
            middle = bound.clip((low + high) / 2)
            if middle in (low, high):  # continuous interval or integral grid exhausted
                break
            probability = float(model.probabilities([{**refined, name: middle}])[0])
            if probability <= target:
                high, score = middle, probability
            else:
                low = middle
        refined[name] = high
    return refined, score


# ---- explainer -------------------------------------------------------------


class CounterfactualExplainer:
    """Recourse for one customer, with a fast fallback when the main search gives up."""

    def __init__(
        self,
        scorer: CreditScorer | None = None,
        search: CounterfactualSearch | None = None,
        fallback: CounterfactualSearch | None = None,
        consistency=None,
        timeout_seconds: float = TIMEOUT_SECONDS,
    ):
        self.scorer = scorer or CreditScorer()
        self.model = ScorerModel(self.scorer)
        self.search = search or GreedySearch()
        self.fallback = fallback or RuleBasedSearch()
        self.consistency = consistency or ArrearsConsistency()
        self.timeout_seconds = timeout_seconds

    def explain(
        self,
        customer: Mapping[str, float],
        target: RecourseTarget | None = None,
        policy: RecoursePolicy | None = None,
    ) -> Counterfactual:
        """Smallest change to the actionable features that reaches the target band."""
        record = self._validated(customer)
        target = target or RecourseTarget()
        policy = policy or default_policy(record)
        ceiling = target.resolve(self.scorer)
        model = ConsistentModel(self.model, self.consistency, record)

        started = time.perf_counter()
        original = float(self.model.probabilities([record])[0])
        outcome, method, notes = self._run(record, policy, ceiling, model, started)
        if outcome.feasible:
            candidate, probability = tighten(record, outcome.candidate, policy, ceiling, model)
            outcome = SearchOutcome(
                candidate, probability, True, outcome.iterations, outcome.note
            )

        # The searches work on the unrepaired vector; report the one that would be acted on.
        repaired = self.consistency.repair(record, outcome.candidate)
        probability = float(self.model.probabilities([repaired])[0])
        outcome = SearchOutcome(
            repaired, probability, probability <= ceiling, outcome.iterations, outcome.note
        )

        return self._describe(
            record,
            outcome,
            policy,
            original,
            ceiling,
            method,
            notes,
            time.perf_counter() - started,
        )

    def _run(
        self,
        record: dict[str, float],
        policy: RecoursePolicy,
        ceiling: float,
        model: ProbabilityModel,
        started: float,
    ) -> tuple[SearchOutcome, str, list[str]]:
        """Primary search, then the fallback if it fails, times out or raises."""
        deadline = started + self.timeout_seconds
        notes: list[str] = []
        try:
            outcome = self.search.search(record, policy, ceiling, model, deadline)
        except Exception as error:  # a failed strategy must not deny the customer an answer
            outcome = SearchOutcome(dict(record), float("nan"), False, 0, str(error))
            notes.append(f"{self.search.name} failed: {error}")

        if outcome.feasible:
            return outcome, self.search.name, notes

        notes.append(f"{self.search.name}: {outcome.note or 'no counterfactual found'}")
        fallback = self.fallback.search(
            record, policy, ceiling, model, time.perf_counter() + self.timeout_seconds
        )
        # Neither reached the target: keep whichever got closest, so an infeasible
        # answer still shows the best effort rather than an arbitrary one.
        if (
            not fallback.feasible
            and not np.isnan(outcome.probability)
            and outcome.probability <= fallback.probability
        ):
            return outcome, self.search.name, notes

        if fallback.note:
            notes.append(f"{self.fallback.name}: {fallback.note}")
        return fallback, self.fallback.name, notes

    def _validated(self, customer: Mapping[str, float]) -> dict[str, float]:
        missing = [column for column in SCORED_COLUMNS if column not in customer]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return dict(customer)

    def _describe(
        self,
        original_record: Mapping[str, float],
        outcome: SearchOutcome,
        policy: RecoursePolicy,
        original: float,
        ceiling: float,
        method: str,
        notes: list[str],
        elapsed: float,
    ) -> Counterfactual:
        """Turn a search outcome into the structured result the API returns."""
        changes = {
            name: {
                "from": float(original_record[name]),
                "to": float(outcome.candidate[name]),
                "delta": float(outcome.candidate[name]) - float(original_record[name]),
            }
            for name in outcome.candidate
            if abs(float(outcome.candidate[name]) - float(original_record[name]))
            > NEGLIGIBLE_DELTA
        }
        achieved = float(outcome.probability)
        if outcome.feasible and not changes:
            notes.append("already at or below the target; no change required")
        return Counterfactual(
            original_score=round(original, 4),
            target_score=round(ceiling, 4),
            achieved_score=round(achieved, 4) if not np.isnan(achieved) else achieved,
            feature_changes=changes,
            is_feasible=bool(outcome.feasible),
            original_band=self.model.band(original),
            achieved_band=self.model.band(achieved) if not np.isnan(achieved) else "",
            method=method,
            iterations=outcome.iterations,
            elapsed_seconds=round(elapsed, 3),
            total_cost=round(policy.cost(original_record, outcome.candidate), 4),
            notes=notes,
        )


def main() -> None:
    from app.ml.scorer import example_customer

    customer = dict(example_customer())
    customer.update(
        {"PAY_1": 2, "PAY_2": 2, "PAY_3": 1, "PAY_AMT1": 0, "PAY_AMT2": 0, "BILL_AMT1": 195000}
    )

    explainer = CounterfactualExplainer()
    for label, target in (("Low risk", RecourseTarget(band="Low")), ("Medium risk", RecourseTarget(band="Medium"))):
        result = explainer.explain(customer, target)
        print("=" * 76)
        print(f"TARGET: {label}   ({result.method}, {result.elapsed_seconds}s)")
        print("=" * 76)
        print(
            f"  score {result.original_score:.1%} ({result.original_band}) "
            f"-> {result.achieved_score:.1%} ({result.achieved_band}), "
            f"ceiling {result.target_score:.1%}, feasible={result.is_feasible}"
        )
        for name, change in result.feature_changes.items():
            print(f"   {name:12s} {change['from']:>12,.0f} -> {change['to']:>12,.0f}"
                  f"  ({change['delta']:+,.0f})")
        for note in result.notes:
            print(f"   note: {note}")
        print()


if __name__ == "__main__":
    main()
