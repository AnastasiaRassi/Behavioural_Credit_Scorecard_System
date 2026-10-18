"""Clustering studies: which algorithm, and what to cluster on.

Two separate questions, often confused.

**Which algorithm.** K-means assumes round, equally sized clusters. Hierarchical makes no
such assumption and shows nested structure. DBSCAN finds arbitrary shapes and labels
outliers as noise rather than forcing them into a group. A Gaussian mixture is
probabilistic: a customer gets a membership probability for every segment instead of one
hard label, which matters when customers sit on a continuum rather than in crisp groups.

**What to cluster on.** `segmentation.py` clusters on *behaviour* — how customers use and
repay credit. This module adds clustering on the *model's outputs*: the SHAP contribution
vector for each customer. That groups customers by **why the model thinks they are risky**,
not by how risky they are. Two customers can both score 0.6 for entirely different reasons,
and only output-based segmentation separates them.

Output-based segmentation feeds the project's research angle directly. If a reason-archetype
is coherent, customers inside it should receive similar explanations; where they do not, the
explanation method is inconsistent.

Run `python -m app.ml.clustering_comparison` to regenerate `clustering_comparison.json`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from app.ml.feature_engineering import build_splits, project_root
from app.ml.scorer import CreditScorer
from app.ml.segmentation import SEGMENT_FEATURES, prepare

SEED = 42
SILHOUETTE_SAMPLE = 5000
# Agglomerative clustering builds an n x n distance matrix; 21,000 rows would need
# roughly 3.5 GB. Fit on a subsample and assign the rest by nearest centroid.
HIERARCHICAL_SUBSAMPLE = 6000
CANDIDATE_K = range(2, 7)


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def quality(matrix: np.ndarray, labels: np.ndarray, target: pd.Series | None = None) -> dict:
    """Internal cluster quality, plus the practical measure: does risk separate?"""
    real = labels[labels >= 0]
    mask = labels >= 0
    distinct = len(set(real))
    if distinct < 2:
        return {"clusters": distinct, "note": "degenerate, fewer than two clusters"}

    scores = {
        "clusters": int(distinct),
        "noise_share": float((labels < 0).mean()),
        "silhouette": float(
            silhouette_score(matrix[mask], real, sample_size=min(SILHOUETTE_SAMPLE, mask.sum()),
                             random_state=SEED)
        ),
        # Lower is better; ratio of within-cluster scatter to between-cluster separation.
        "davies_bouldin": float(davies_bouldin_score(matrix[mask], real)),
        # Higher is better; variance ratio.
        "calinski_harabasz": float(calinski_harabasz_score(matrix[mask], real)),
    }
    if target is not None:
        rates = pd.Series(target.to_numpy()[mask]).groupby(real).mean()
        scores["default_rate_spread"] = float(rates.max() - rates.min())
        scores["default_rates"] = {str(k): round(float(v), 4) for k, v in rates.items()}
        sizes = pd.Series(real).value_counts(normalize=True)
        scores["smallest_cluster_share"] = float(sizes.min())
    return scores


def compare_algorithms(scaled_train, scaled_eval, y_eval) -> list[dict]:
    """Four algorithm families on the same behavioural features."""
    print("\n[1/2] Clustering algorithms on behavioural features")
    results = []

    def record(name, labels, seconds, note=""):
        row = {"algorithm": name, "seconds": round(seconds, 1), "note": note}
        row.update(quality(scaled_eval, labels, y_eval))
        results.append(row)
        if "silhouette" in row:
            print(f"  {name:32s} k={row['clusters']}  sil={row['silhouette']:.3f}  "
                  f"DB={row['davies_bouldin']:.2f}  spread={row.get('default_rate_spread', 0):.3f}"
                  f"  {row['seconds']:5.1f}s")
        else:
            print(f"  {name:32s} {row['note']}")

    for k in CANDIDATE_K:
        start = time.time()
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled_train)
        record(f"KMeans k={k}", model.predict(scaled_eval), time.time() - start,
               "spherical clusters of similar size")

    for k in CANDIDATE_K:
        start = time.time()
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED,
                                max_iter=200).fit(scaled_train)
        record(f"GaussianMixture k={k}", model.predict(scaled_eval), time.time() - start,
               "probabilistic, soft membership, elliptical clusters")

    rng = np.random.default_rng(SEED)
    subset = rng.choice(len(scaled_train), min(HIERARCHICAL_SUBSAMPLE, len(scaled_train)),
                        replace=False)
    for linkage in ("ward", "average"):
        for k in (3, 4, 5):
            start = time.time()
            fitted = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(
                scaled_train[subset]
            )
            # Assign held-out rows to the nearest cluster centroid.
            centroids = np.array(
                [scaled_train[subset][fitted.labels_ == c].mean(axis=0) for c in range(k)]
            )
            labels = np.argmin(
                ((scaled_eval[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
            )
            record(f"Hierarchical {linkage} k={k}", labels, time.time() - start,
                   f"fitted on {len(subset):,} rows, assigned by nearest centroid")

    for eps in (0.5, 0.8, 1.2, 1.8):
        start = time.time()
        model = DBSCAN(eps=eps, min_samples=50, n_jobs=-1).fit(scaled_eval)
        record(f"DBSCAN eps={eps}", model.labels_, time.time() - start,
               "density based; -1 means noise, and k is discovered not chosen")

    return results


def segment_on_outputs(scorer: CreditScorer, frames, y) -> dict:
    """Cluster on SHAP vectors: group customers by *why* the model scores them as it does."""
    print("\n[2/2] Segmenting on classification outputs (SHAP space)")

    def contributions(frame):
        prepared = scorer.prepare(frame)
        stacked = [m.predict(prepared, pred_contrib=True) for m in scorer.models]
        # Drop the base value column; it is identical for every customer.
        return np.mean(stacked, axis=0)[:, :-1]

    train_shap = contributions(frames["train"])
    test_shap = contributions(frames["test"])
    print(f"  SHAP matrices: train {train_shap.shape}, test {test_shap.shape}")

    scaler = StandardScaler().fit(train_shap)
    train_scaled, test_scaled = scaler.transform(train_shap), scaler.transform(test_shap)

    candidates = {}
    for k in CANDIDATE_K:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(train_scaled)
        labels = model.predict(test_scaled)
        candidates[k] = {
            "model": model,
            "labels": labels,
            "quality": quality(test_scaled, labels, y["test"]),
        }
        q = candidates[k]["quality"]
        print(f"  k={k}  sil={q['silhouette']:.3f}  "
              f"default spread={q.get('default_rate_spread', 0):.3f}")

    best_k = max(candidates, key=lambda k: candidates[k]["quality"]["silhouette"])
    labels = candidates[best_k]["labels"]
    scores = scorer.probabilities(frames["test"])

    # Name each reason-archetype by the features that drive it.
    feature_names = scorer.feature_names
    profiles = []
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        mean_contribution = test_shap[mask].mean(axis=0)
        order = np.argsort(-np.abs(mean_contribution))[:4]
        drivers = [
            {
                "feature": feature_names[i],
                "mean_contribution": round(float(mean_contribution[i]), 4),
                "direction": "raises risk" if mean_contribution[i] > 0 else "lowers risk",
            }
            for i in order
        ]
        profiles.append({
            "cluster": int(cluster),
            "customers": int(mask.sum()),
            "share": round(float(mask.mean()), 4),
            "default_rate": round(float(y["test"].to_numpy()[mask].mean()), 4),
            "mean_predicted": round(float(scores[mask].mean()), 4),
            "dominant_reasons": drivers,
        })

    print(f"\n  chosen k={best_k}")
    print(f"  {'cluster':>7s} {'n':>6s} {'default':>8s} {'predicted':>10s}  dominant reason")
    for p in profiles:
        top = p["dominant_reasons"][0]
        print(f"  {p['cluster']:7d} {p['customers']:6d} {p['default_rate']:8.1%} "
              f"{p['mean_predicted']:10.1%}  {top['feature']} {top['direction']}")

    return {
        "clustered_on": "mean SHAP contribution vector across the seed ensemble",
        "chosen_k": int(best_k),
        "k_candidates": {
            str(k): candidates[k]["quality"] for k in candidates
        },
        "reason_archetypes": profiles,
    }


def main() -> dict:
    frames, y, _ = build_splits(seed=SEED, blind=True)

    train_prepared = prepare(frames["train"])
    eval_prepared = prepare(frames["test"])
    imputer = SimpleImputer(strategy="median").fit(train_prepared)
    scaler = StandardScaler().fit(imputer.transform(train_prepared))
    scaled_train = scaler.transform(imputer.transform(train_prepared))
    scaled_eval = scaler.transform(imputer.transform(eval_prepared))

    print(f"Behavioural features: {len(SEGMENT_FEATURES)}; "
          f"train {len(scaled_train):,}, test {len(scaled_eval):,}")

    algorithms = compare_algorithms(scaled_train, scaled_eval, y["test"])
    outputs = segment_on_outputs(CreditScorer(), frames, y)

    payload = {
        "seed": SEED,
        "behavioural_features": SEGMENT_FEATURES,
        "evaluated_on": "test split",
        "algorithm_comparison": algorithms,
        "output_based_segmentation": outputs,
    }
    out = artifacts_dir() / "clustering_comparison.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    usable = [a for a in algorithms if "silhouette" in a]
    best = max(usable, key=lambda a: a["silhouette"])
    print("\n" + "=" * 72)
    print(f"Best by silhouette: {best['algorithm']} ({best['silhouette']:.3f})")
    widest = max(usable, key=lambda a: a.get("default_rate_spread", 0))
    print(f"Widest risk separation: {widest['algorithm']} "
          f"({widest.get('default_rate_spread', 0):.3f})")
    print(f"Written to {out}")
    return payload


if __name__ == "__main__":
    main()
