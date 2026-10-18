"""Predictive multiplicity: when equally accurate models disagree about individuals.

The model comparison study found roughly a dozen configurations within noise of each
other on AUC. Aggregate equivalence is not individual equivalence: two models with the
same accuracy can flag different customers and, for the customers they both flag, give
different reasons.

That matters here specifically. Under a right to explanation, the reasons a lender gives
are part of the decision. If an arbitrary choice between two equally defensible models
changes who is declined and what they are told, then "the model said so" is not a
sufficient justification, and the choice of model becomes a policy decision rather than a
technical one.

Three questions, in order:

1. **Decisions.** What share of customers does one equally good model flag while another
   does not?
2. **Reasons.** For customers every model flags, do they all give the same top three
   drivers?
3. **Who it affects.** Are the disagreements spread evenly, or concentrated in particular
   behavioural segments or demographic groups?

Two measures from the literature are reported. **Ambiguity** is the share of customers who
receive conflicting decisions anywhere in the model set. **Discrepancy** is the largest
share of decisions that change between the deployed model and any single competitor.

Models are compared at a **matched flag rate**: each flags the same number of customers,
chosen as its own top-k by score. Without that, differences in threshold calibration would
masquerade as disagreement.

Run `python -m app.ml.multiplicity` to regenerate `multiplicity.json`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.feature_engineering import PROTECTED, build_splits, project_root
from app.ml.segmentation import Segmenter
from app.ml.train import HYPERPARAMETERS

SEED = 42
# Customers sampled for the reasons study. Occlusion costs one prediction per feature
# per model, so this is the dial that controls runtime.
REASON_SAMPLE = 250
TOP_K_REASONS = 3
AGE_BANDS = [20, 30, 40, 50, 60, 100]
AGE_LABELS = ["21-30", "31-40", "41-50", "51-60", "60+"]


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def imputed(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def scaled(model):
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler()),
         ("model", model)]
    )


def candidate_models() -> dict:
    """Configurations the comparison study found statistically indistinguishable."""
    return {
        "LightGBM (deployed)": LGBMClassifier(
            **HYPERPARAMETERS, n_jobs=-1, random_state=SEED, verbose=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=5, min_child_weight=5,
            subsample=0.9, colsample_bytree=0.8, tree_method="hist",
            eval_metric="auc", n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=SEED),
        "RandomForest": imputed(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "ExtraTrees": imputed(ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "LightGBM (seed 777)": LGBMClassifier(
            **HYPERPARAMETERS, n_jobs=-1, random_state=777, verbose=-1),
    }


def flags_at_matched_rate(scores: np.ndarray, n_flagged: int) -> np.ndarray:
    """Flag the top n_flagged customers by score, so every model flags the same number."""
    cutoff = np.partition(scores, -n_flagged)[-n_flagged]
    return scores >= cutoff


def occlusion_reasons(model, features: pd.DataFrame, medians: pd.Series,
                      k: int = TOP_K_REASONS) -> list[list[str]]:
    """Top-k drivers per customer, by median substitution.

    Model-agnostic on purpose: using each model's native explainer would confound
    disagreement between models with disagreement between explanation methods.
    """
    names = list(features.columns)
    base = model.predict_proba(features)[:, 1]

    # One stacked frame per feature rather than a loop of single predictions.
    effects = np.empty((len(features), len(names)))
    for position, name in enumerate(names):
        altered = features.astype("float64").copy()
        altered[name] = float(medians[name])
        effects[:, position] = base - model.predict_proba(altered)[:, 1]

    order = np.argsort(-np.abs(effects), axis=1)[:, :k]
    return [[names[i] for i in row] for row in order]


def jaccard(left, right) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def main() -> dict:
    frames, y, audit = build_splits(seed=SEED, blind=True)
    X_dev = pd.concat([frames["train"], frames["validation"]])
    y_dev = pd.concat([y["train"], y["validation"]])
    X_test, y_test = frames["test"], y["test"]

    print(f"Fitting {len(candidate_models())} models on {len(X_dev):,} rows, "
          f"comparing on {len(X_test):,} held-out customers.\n")

    scores, aucs, fitted = {}, {}, {}
    for name, model in candidate_models().items():
        start = time.time()
        model.fit(X_dev, y_dev)
        scores[name] = model.predict_proba(X_test)[:, 1]
        aucs[name] = float(roc_auc_score(y_test, scores[name]))
        print(f"  {name:24s} auc={aucs[name]:.4f}  {time.time()-start:5.1f}s")
        fitted[name] = model

    names = list(scores)
    spread = max(aucs.values()) - min(aucs.values())
    print(f"\nAUC spread across the set: {spread:.4f}")

    # ---------------------------------------------------- 1. decisions
    # A quarter of the book, matched across models so threshold differences
    # cannot masquerade as disagreement about individuals.
    n_flagged = int(round(0.25 * len(X_test)))
    flags = {name: flags_at_matched_rate(scores[name], n_flagged) for name in names}
    print(f"\nEach model flags {n_flagged:,} customers ({n_flagged/len(X_test):.0%}).")

    stacked = np.vstack([flags[n] for n in names])
    agreed_flag = stacked.all(axis=0)
    agreed_clear = (~stacked).all(axis=0)
    ambiguous = ~(agreed_flag | agreed_clear)

    pairwise = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            disagreement = float((flags[left] != flags[right]).mean())
            pairwise.append({"a": left, "b": right, "disagreement": round(disagreement, 4)})

    baseline = names[0]
    discrepancy = max(float((flags[baseline] != flags[n]).mean()) for n in names[1:])

    print("\n=== 1. DECISIONS ===")
    print(f"  flagged by every model : {agreed_flag.sum():,} ({agreed_flag.mean():.1%})")
    print(f"  cleared by every model : {agreed_clear.sum():,} ({agreed_clear.mean():.1%})")
    print(f"  AMBIGUITY (conflicting): {ambiguous.sum():,} ({ambiguous.mean():.1%})")
    print(f"  DISCREPANCY vs deployed: {discrepancy:.1%}")
    worst = max(pairwise, key=lambda r: r["disagreement"])
    print(f"  widest pair            : {worst['a']} vs {worst['b']} "
          f"({worst['disagreement']:.1%})")

    # ---------------------------------------------------- 2. reasons
    print("\n=== 2. REASONS ===")
    medians = X_dev.median()
    candidates = np.where(agreed_flag)[0]
    rng = np.random.default_rng(SEED)
    sample = rng.choice(candidates, min(REASON_SAMPLE, len(candidates)), replace=False)
    sample_features = X_test.iloc[sample]
    print(f"  {len(sample)} customers every model flags; top-{TOP_K_REASONS} drivers each")

    reasons = {}
    for name in names:
        start = time.time()
        reasons[name] = occlusion_reasons(fitted[name], sample_features, medians)
        print(f"    {name:24s} {time.time()-start:5.1f}s")

    per_customer = []
    for position in range(len(sample)):
        sets = [reasons[n][position] for n in names]
        overlaps = [
            jaccard(sets[i], sets[j])
            for i in range(len(sets)) for j in range(i + 1, len(sets))
        ]
        per_customer.append({
            "mean_pairwise_jaccard": float(np.mean(overlaps)),
            "identical_across_all": len({tuple(sorted(s)) for s in sets}) == 1,
            "distinct_reason_sets": len({tuple(sorted(s)) for s in sets}),
            "top_driver_unanimous": len({s[0] for s in sets}) == 1,
        })

    reasons_frame = pd.DataFrame(per_customer)
    print(f"\n  identical top-{TOP_K_REASONS} from all models : "
          f"{reasons_frame['identical_across_all'].mean():.1%} of customers")
    print(f"  same single top driver            : "
          f"{reasons_frame['top_driver_unanimous'].mean():.1%} of customers")
    print(f"  mean pairwise overlap             : "
          f"{reasons_frame['mean_pairwise_jaccard'].mean():.3f}")
    print(f"  median distinct reason sets       : "
          f"{reasons_frame['distinct_reason_sets'].median():.0f} of {len(names)}")

    # ---------------------------------------------------- 3. who it affects
    print("\n=== 3. WHO IT AFFECTS ===")
    segments = Segmenter().assign(X_test)["segment_name"].to_numpy()
    groups = {
        "segment": segments,
        "SEX": audit["test"]["SEX"].map({1: "male", 2: "female"}).to_numpy(),
        "EDUCATION": audit["test"]["EDUCATION"].map(
            {1: "graduate", 2: "university", 3: "high school", 4: "other"}).to_numpy(),
        "AGE": pd.cut(audit["test"]["AGE"], AGE_BANDS, labels=AGE_LABELS).astype(str).to_numpy(),
    }

    concentration = {}
    overall = ambiguous.mean()
    for label, values in groups.items():
        rows = []
        for value in pd.unique(values):
            mask = values == value
            if mask.sum() < 50:
                continue
            rows.append({
                "group": str(value),
                "customers": int(mask.sum()),
                "ambiguous_rate": round(float(ambiguous[mask].mean()), 4),
                "lift_vs_overall": round(float(ambiguous[mask].mean() / overall), 2)
                if overall else None,
                "default_rate": round(float(y_test.to_numpy()[mask].mean()), 4),
            })
        frame = pd.DataFrame(rows).sort_values("ambiguous_rate", ascending=False)
        concentration[label] = frame.to_dict(orient="records")
        print(f"\n  by {label}:")
        print("   " + frame.to_string(index=False).replace("\n", "\n   "))

    payload = {
        "seed": SEED,
        "models": {name: round(aucs[name], 4) for name in names},
        "auc_spread": round(spread, 4),
        "matched_flag_rate": round(n_flagged / len(X_test), 4),
        "decisions": {
            "flagged_by_all": int(agreed_flag.sum()),
            "cleared_by_all": int(agreed_clear.sum()),
            "ambiguity": round(float(ambiguous.mean()), 4),
            "discrepancy_vs_deployed": round(discrepancy, 4),
            "pairwise_disagreement": pairwise,
        },
        "reasons": {
            "customers_examined": int(len(sample)),
            "top_k": TOP_K_REASONS,
            "identical_across_all_models": round(
                float(reasons_frame["identical_across_all"].mean()), 4),
            "top_driver_unanimous": round(
                float(reasons_frame["top_driver_unanimous"].mean()), 4),
            "mean_pairwise_jaccard": round(
                float(reasons_frame["mean_pairwise_jaccard"].mean()), 4),
        },
        "concentration": concentration,
    }

    out = artifacts_dir() / "multiplicity.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWritten to {out}")
    return payload


if __name__ == "__main__":
    main()
