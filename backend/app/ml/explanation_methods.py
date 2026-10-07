"""Three explanation methods behind one interface, so they can be compared.

The project's research angle is whether explanation methods are *consistent*. That needs
more than one method, and it needs them to return comparable output. Each method here
answers the same question — which features drove this customer's score, and in which
direction — and returns the same shape.

| Method | How it works | Deterministic |
| --- | --- | --- |
| `TreeShap` | Reads the tree structure directly | Yes, exact |
| `Lime` | Samples around the customer, fits a weighted linear model | **No**, samples randomly |
| `Occlusion` | Replaces one feature with its median, measures the change | Yes |

That middle row is the point. LIME draws a random sample every call, so the same customer
can receive a different explanation on a second run. Tree SHAP cannot, because it is
computed from the model rather than estimated from samples. Quantifying that difference is
the experiment.

One practical asymmetry worth reporting: LIME requires finite inputs, so the engineered
features have to be median-imputed before it runs. LightGBM handles missing values
natively, so SHAP and occlusion explain the customer as the model actually sees them.
LIME therefore explains a slightly different customer, which is a cost of the method rather
than a bug in the setup.

Run `python -m app.ml.explanation_methods` for a demonstration on one customer.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from app.ml.feature_engineering import build_splits
from app.ml.scorer import CreditScorer

SEED = 42
LIME_SAMPLES = 5000


@dataclass
class Explanation:
    """One method's account of one customer's score."""

    method: str
    contributions: dict[str, float]
    probability: float

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        ordered = sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))
        return ordered[:k]

    def top_features(self, k: int = 5) -> list[str]:
        return [feature for feature, _ in self.top(k)]

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _background():
    """Training features and a median imputer, shared by the methods that need them."""
    frames, _, _ = build_splits(seed=SEED, blind=True)
    train = frames["train"]
    imputer = SimpleImputer(strategy="median").fit(train)
    return train, imputer


class ExplanationMethod:
    """Base class. Subclasses implement `explain` for a single prepared row."""

    name = "base"

    def __init__(self, scorer: CreditScorer | None = None):
        self.scorer = scorer or CreditScorer()
        self.feature_names = self.scorer.feature_names

    def _prepared(self, customer: dict) -> pd.DataFrame:
        return self.scorer.prepare(pd.DataFrame([customer]))

    def _probability(self, features: pd.DataFrame) -> float:
        raw = np.mean([m.predict_proba(features)[:, 1] for m in self.scorer.models], axis=0)
        return float(np.clip(self.scorer.calibrator.predict(raw), 0, 1)[0])

    def explain(self, customer: dict, seed: int | None = None) -> Explanation:
        raise NotImplementedError


class TreeShap(ExplanationMethod):
    """Exact SHAP from LightGBM's own tree traversal. No sampling, no approximation."""

    name = "TreeSHAP"

    def explain(self, customer: dict, seed: int | None = None) -> Explanation:
        features = self._prepared(customer)
        # Last column is the base value; the rest are per-feature contributions in log odds.
        stacked = [m.predict(features, pred_contrib=True) for m in self.scorer.models]
        values = np.mean(stacked, axis=0)[0][:-1]
        return Explanation(
            method=self.name,
            contributions={n: float(v) for n, v in zip(self.feature_names, values)},
            probability=self._probability(features),
        )


class Lime(ExplanationMethod):
    """LIME: perturb around the customer, weight by proximity, fit a linear surrogate.

    `seed` is exposed deliberately. Left unset, repeated calls draw different samples and
    return different explanations, which is exactly what the stability experiment measures.
    """

    name = "LIME"

    def __init__(self, scorer: CreditScorer | None = None, num_samples: int = LIME_SAMPLES):
        super().__init__(scorer)
        from lime.lime_tabular import LimeTabularExplainer

        train, imputer = _background()
        self.imputer = imputer
        self.num_samples = num_samples
        self._explainer_class = LimeTabularExplainer
        self._training = imputer.transform(train)

    def _predict(self, array: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(array, columns=self.feature_names)
        raw = np.mean([m.predict_proba(frame)[:, 1] for m in self.scorer.models], axis=0)
        calibrated = np.clip(self.scorer.calibrator.predict(raw), 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

    def explain(self, customer: dict, seed: int | None = None) -> Explanation:
        features = self._prepared(customer)
        row = self.imputer.transform(features)[0]

        explainer = self._explainer_class(
            self._training,
            feature_names=self.feature_names,
            class_names=["pays", "defaults"],
            mode="classification",
            discretize_continuous=True,
            random_state=seed,
        )
        result = explainer.explain_instance(
            row,
            self._predict,
            num_features=len(self.feature_names),
            num_samples=self.num_samples,
        )

        # LIME labels conditions ("pay_max > 1.00"); map each back to its feature.
        contributions = {name: 0.0 for name in self.feature_names}
        for index, weight in result.as_map()[1]:
            contributions[self.feature_names[index]] = float(weight)

        return Explanation(
            method=self.name,
            contributions=contributions,
            probability=self._probability(features),
        )


class Occlusion(ExplanationMethod):
    """Replace one feature with its training median and measure the change in score.

    A deliberately simple third opinion. It needs no library and makes no assumption about
    local linearity, so where it disagrees with both other methods that is informative.
    """

    name = "Occlusion"

    def __init__(self, scorer: CreditScorer | None = None):
        super().__init__(scorer)
        train, _ = _background()
        self.medians = train.median()

    def explain(self, customer: dict, seed: int | None = None) -> Explanation:
        features = self._prepared(customer)
        baseline = self._probability(features)

        # One batch rather than 46 calls: each row has a single feature replaced.
        occluded = pd.concat([features] * len(self.feature_names), ignore_index=True)
        for position, name in enumerate(self.feature_names):
            occluded.loc[position, name] = self.medians[name]

        raw = np.mean([m.predict_proba(occluded)[:, 1] for m in self.scorer.models], axis=0)
        shifted = np.clip(self.scorer.calibrator.predict(raw), 0, 1)

        return Explanation(
            method=self.name,
            # Positive means removing the feature lowers the score, so it was raising risk.
            contributions={
                name: float(baseline - shifted[position])
                for position, name in enumerate(self.feature_names)
            },
            probability=baseline,
        )


def jaccard(left: list[str], right: list[str]) -> float:
    """Overlap between two top-k feature sets."""
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def run_to_run_stability(
    method: ExplanationMethod, customer: dict, runs: int = 10, k: int = 5
) -> dict:
    """Explain the same customer repeatedly and measure how much the answer moves.

    No seed is passed, so any sampling the method does is free to vary. A deterministic
    method scores 1.0 on every measure; a sampling one will not.
    """
    explanations = [method.explain(customer) for _ in range(runs)]
    top_sets = [e.top_features(k) for e in explanations]

    pairwise = [
        jaccard(top_sets[i], top_sets[j])
        for i in range(len(top_sets))
        for j in range(i + 1, len(top_sets))
    ]
    matrix = np.array([[e.contributions[n] for n in method.feature_names] for e in explanations])
    identical = len({tuple(s) for s in top_sets}) == 1

    return {
        "method": method.name,
        "runs": runs,
        "mean_top_k_jaccard": float(np.mean(pairwise)),
        "identical_top_k_every_run": identical,
        "distinct_top_k_orderings": len({tuple(s) for s in top_sets}),
        "max_contribution_sd": float(matrix.std(axis=0).max()),
    }


def main() -> None:
    from app.ml.scorer import example_customer

    scorer = CreditScorer()
    customer = dict(example_customer())
    customer.update(
        {"PAY_1": 2, "PAY_2": 2, "PAY_3": 1, "PAY_AMT1": 0, "PAY_AMT2": 0, "BILL_AMT1": 195000}
    )

    methods = [TreeShap(scorer), Lime(scorer), Occlusion(scorer)]

    print("=" * 76)
    print("TOP 5 FEATURES BY METHOD (same customer)")
    print("=" * 76)
    explanations = {}
    for method in methods:
        explanation = method.explain(customer)
        explanations[method.name] = explanation
        print(f"\n{method.name}  (probability {explanation.probability:.1%})")
        for feature, value in explanation.top(5):
            print(f"   {feature:24s} {value:+.4f}")

    print("\n" + "=" * 76)
    print("AGREEMENT BETWEEN METHODS (top-5 Jaccard)")
    print("=" * 76)
    names = list(explanations)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = jaccard(
                explanations[left].top_features(5), explanations[right].top_features(5)
            )
            print(f"  {left:10s} vs {right:10s}  {overlap:.2f}")

    print("\n" + "=" * 76)
    print("RUN-TO-RUN STABILITY (10 runs, same customer, no seed fixed)")
    print("=" * 76)
    for method in methods:
        stability = run_to_run_stability(method, customer, runs=10)
        verdict = "stable" if stability["identical_top_k_every_run"] else "VARIES"
        print(
            f"  {stability['method']:10s} jaccard={stability['mean_top_k_jaccard']:.3f}  "
            f"distinct top-5 sets={stability['distinct_top_k_orderings']:2d}  "
            f"max sd={stability['max_contribution_sd']:.4f}  [{verdict}]"
        )


if __name__ == "__main__":
    main()
