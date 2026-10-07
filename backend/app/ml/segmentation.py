"""Behavioural segmentation of credit-card customers.

Groups customers by how they use and repay credit, independently of the risk model.
Card issuers segment this way in practice: a customer who clears the balance every month
is a different proposition from one who revolves at the limit, even at equal risk.

Two uses in this project:

1. The dashboard can report performance and fairness per segment rather than only in
   aggregate, which is where a global average hides a problem.
2. It supports the research angle. Explanations should be coherent *within* a segment:
   if two near-identical revolvers receive different top reasons, the explanation method
   is unstable in a way a single-customer test cannot reveal.

Clustering uses a compact, interpretable subset of the engineered features rather than all
46. The monthly columns are strongly correlated, and k-means on the full set would be
dominated by whichever block has the most columns instead of by behaviour.

The target and the protected attributes are excluded, so segments describe behaviour and
can be audited for fairness without circularity. Scaler, imputer and cluster centres are
fitted on the training split only.

Run `python -m app.ml.segmentation` from `backend/` to regenerate the artifacts.
"""
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.ml.feature_engineering import PROTECTED, build_splits, project_root

SEED = 42
CANDIDATE_K = range(2, 9)
# silhouette_score is O(n^2); sample rather than scoring all 21,000 training rows.
SILHOUETTE_SAMPLE = 5000

# Compact behavioural profile. Three blocks, no block allowed to dominate by column count.
SEGMENT_FEATURES = [
    # delinquency
    "pay_max",
    "pay_months_late",
    "pay_trend",
    # utilisation
    "util_mean",
    "util_trend",
    # repayment behaviour
    "repay_ratio_mean",
    "months_zero_payment",
    # scale of the relationship
    "LIMIT_BAL",
    "bill_mean",
]

# Money columns are heavily right-skewed; log1p keeps one large balance from defining a
# cluster on its own. signed log handles the negative balances that appear in BILL_AMT.
LOG_FEATURES = ["LIMIT_BAL", "bill_mean"]


@dataclass
class SegmentProfile:
    label: int
    name: str
    customers: int
    share: float
    default_rate: float
    distinctive: list[str]
    means: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class ArtifactsMissing(RuntimeError):
    pass


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _signed_log(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(np.abs(values))


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the behavioural subset and tame the skew. No fitted statistics here."""
    missing = [c for c in SEGMENT_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Segment features absent from the frame: {missing}")

    out = frame[SEGMENT_FEATURES].copy()
    for column in LOG_FEATURES:
        out[column] = _signed_log(out[column])
    return out


def choose_k(scaled: np.ndarray, seed: int = SEED) -> tuple[int, dict[int, float]]:
    """Pick k by silhouette. Returns the winner and every score, so the choice is auditable."""
    scores = {}
    for k in CANDIDATE_K:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scaled)
        scores[k] = float(
            silhouette_score(
                scaled, labels, sample_size=SILHOUETTE_SAMPLE, random_state=seed
            )
        )
    return max(scores, key=scores.get), scores


def name_segment(means: pd.Series, overall: pd.Series) -> str:
    """Heuristic display label from the segment's behaviour.

    Deliberately simple and rule-based: these are labels for a screen, not claims. The
    numeric profile beside them is the evidence.
    """
    late = means["pay_months_late"]
    utilisation = means["util_mean"]
    repayment = means["repay_ratio_mean"]
    unpaid = means["months_zero_payment"]

    if late >= 1.5:
        return "Distressed"
    if unpaid >= 3 or (utilisation < 0.05 and means["bill_mean"] < overall["bill_mean"]):
        return "Dormant"
    if repayment >= 0.8 and utilisation < overall["util_mean"]:
        return "Transactor"
    if utilisation >= overall["util_mean"] and repayment < 0.3:
        return "Revolver"
    return "Moderate user"


def profile(
    features: pd.DataFrame, labels: np.ndarray, target: pd.Series
) -> list[SegmentProfile]:
    """Describe each segment: size, risk, and what makes it distinctive."""
    overall = features.mean()
    spread = features.std().replace(0, np.nan)

    profiles = []
    for label in sorted(set(labels)):
        mask = labels == label
        means = features.loc[mask].mean()

        # Distinctive = furthest from the overall mean in standard deviations.
        distance = ((means - overall) / spread).abs().sort_values(ascending=False)
        distinctive = [
            f"{feature} {'high' if means[feature] > overall[feature] else 'low'}"
            for feature in distance.head(3).index
        ]

        profiles.append(
            SegmentProfile(
                label=int(label),
                name=name_segment(means, overall),
                customers=int(mask.sum()),
                share=float(mask.mean()),
                default_rate=float(target.to_numpy()[mask].mean()),
                distinctive=distinctive,
                means={k: round(float(v), 4) for k, v in means.items()},
            )
        )
    return profiles


@lru_cache(maxsize=1)
def _load():
    path = artifacts_dir()
    model_path = path / "segmenter.joblib"
    config_path = path / "segments.json"
    missing = [p.name for p in (model_path, config_path) if not p.exists()]
    if missing:
        raise ArtifactsMissing(
            f"Missing artifacts {missing} in {path}. Run: python -m app.ml.segmentation"
        )
    with open(config_path, encoding="utf-8") as fh:
        return joblib.load(model_path), json.load(fh)


class Segmenter:
    """Assigns customers to a behavioural segment. Cheap to construct."""

    def __init__(self):
        pipeline, config = _load()
        self.imputer = pipeline["imputer"]
        self.scaler = pipeline["scaler"]
        self.kmeans = pipeline["kmeans"]
        self.config = config
        self.names = {int(p["label"]): p["name"] for p in config["segments"]}

    def labels(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = prepare(frame)
        return self.kmeans.predict(self.scaler.transform(self.imputer.transform(prepared)))

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        labels = self.labels(frame)
        return pd.DataFrame(
            {"segment": labels, "segment_name": [self.names[int(l)] for l in labels]},
            index=frame.index,
        )


def main() -> dict:
    frames, y, audit = build_splits(seed=SEED, blind=True)

    # Fitted on training rows only, exactly as the model's preprocessing is.
    train_prepared = prepare(frames["train"])
    imputer = SimpleImputer(strategy="median").fit(train_prepared)
    scaler = StandardScaler().fit(imputer.transform(train_prepared))
    scaled_train = scaler.transform(imputer.transform(train_prepared))

    k, silhouettes = choose_k(scaled_train)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled_train)

    # Report on the test split: segment risk should hold on data nobody tuned against.
    test_prepared = prepare(frames["test"])
    test_labels = kmeans.predict(scaler.transform(imputer.transform(test_prepared)))
    profiles = profile(test_prepared, test_labels, y["test"])

    # Protected-group mix per segment, so a segment that is a demographic proxy is visible.
    test_audit = audit["test"].copy()
    test_audit["segment"] = test_labels
    composition = {
        attribute: {
            str(label): {
                str(group): round(float(share), 4)
                for group, share in (
                    test_audit.loc[test_audit["segment"] == label, attribute]
                    .value_counts(normalize=True)
                    .items()
                )
            }
            for label in sorted(set(test_labels))
        }
        for attribute in PROTECTED
        if attribute != "AGE"
    }

    config = {
        "k": int(k),
        "chosen_by": "silhouette score on the training split",
        "silhouette_scores": {str(key): round(value, 4) for key, value in silhouettes.items()},
        "features": SEGMENT_FEATURES,
        "log_transformed": LOG_FEATURES,
        "seed": SEED,
        "fitted_on": "train split only",
        "profiled_on": "test split",
        "segments": [p.to_dict() for p in profiles],
        "protected_composition": composition,
    }

    out = artifacts_dir()
    joblib.dump({"imputer": imputer, "scaler": scaler, "kmeans": kmeans}, out / "segmenter.joblib")
    with open(out / "segments.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print(f"k = {k} chosen by silhouette from {dict(silhouettes)}")
    print(f"\n{'segment':16s} {'n':>6s} {'share':>7s} {'default':>8s}  distinctive")
    print("-" * 78)
    for p in profiles:
        print(
            f"{p.name:16s} {p.customers:6d} {p.share:6.1%} {p.default_rate:8.1%}  "
            + ", ".join(p.distinctive)
        )
    print(f"\nArtifacts written to {out}")
    return config


if __name__ == "__main__":
    main()
