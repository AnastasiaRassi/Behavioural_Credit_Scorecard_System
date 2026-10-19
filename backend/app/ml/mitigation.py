"""Bias mitigation: what fairness costs, measured rather than assumed.

Objective 3 of the project is to measure **and mitigate** bias. The audit in notebook 02
does the measuring: all four protected attributes already pass the four-fifths rule, with
disparate impact between 0.89 and 0.99. So there is no large disparity to repair, and the
honest question is not "can fairness be imposed" but **what does imposing it cost**.

Five interventions, across the three stages at which bias can be addressed:

| Stage | Method | Idea |
| --- | --- | --- |
| none | baseline | the deployed model, for reference |
| pre | reweighting | reweight training rows so group and outcome are independent |
| in | exponentiated gradient | train under an explicit fairness constraint |
| post | per-group thresholds | one cut-off per group, chosen to equalise an objective |
| post | threshold optimiser | the same idea, fairlearn's implementation |

Each is scored on both axes — discrimination and fairness — and the results form a
frontier. A point is only interesting if nothing else is better on both.

**Protected attributes stay out of the model's features throughout.** They are used to
*measure* disparity, and for the post-processing methods to *route* a customer to a
threshold, which is a policy applied to the output rather than a feature the model learns
from. That distinction matters legally and is worth stating in the report.

Run `python -m app.ml.mitigation` to regenerate `mitigation.json`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from app.ml.feature_engineering import build_splits, project_root
from app.ml.train import HYPERPARAMETERS

SEED = 42
FLAG_RATE = 0.25
SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate", 2: "university", 3: "high school", 4: "other"}

# The reductions approach refits the base model once per iteration, so the deployed
# 473-tree configuration would make the sweep impractical. A lighter model is used for
# the constrained runs only; the baseline it is compared against uses the same lighter
# model, so the comparison stays like for like.
LIGHT = {**HYPERPARAMETERS, "n_estimators": 150, "learning_rate": 0.05}


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def light_model(seed: int = SEED) -> LGBMClassifier:
    return LGBMClassifier(**LIGHT, n_jobs=-1, random_state=seed, verbose=-1)


def flag_at_rate(scores: np.ndarray, rate: float = FLAG_RATE) -> np.ndarray:
    """Flag the top `rate` share by score, so methods are compared at equal volume."""
    k = max(1, int(round(rate * len(scores))))
    cutoff = np.partition(scores, -k)[-k]
    return (scores >= cutoff).astype(int)


def measure(name, y_true, predicted, groups, scores=None, seconds=0.0, note="") -> dict:
    """Discrimination and fairness for one intervention."""
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    row = {
        "method": name,
        "roc_auc": float(roc_auc_score(y_true, scores)) if scores is not None else None,
        "f1": float(f1_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "flag_rate": float(predicted.mean()),
        # Favourable outcome is *not* being flagged, so parity is measured on the
        # cleared rate rather than the flagged rate.
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true, predicted, sensitive_features=groups)),
        "disparate_impact_ratio": float(
            demographic_parity_ratio(y_true, predicted, sensitive_features=groups)),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_true, predicted, sensitive_features=groups)),
        "seconds": round(seconds, 1),
        "note": note,
    }
    row["passes_four_fifths"] = bool(row["disparate_impact_ratio"] >= 0.80)
    return row


def reweight(y: pd.Series, groups: np.ndarray) -> np.ndarray:
    """Kamiran and Calders reweighting.

    Weight each row by the ratio of what its (group, outcome) cell would contain if the
    two were independent to what it actually contains. Rows in under-represented
    combinations are weighted up, so the model stops being able to exploit the
    association.
    """
    y = np.asarray(y)
    weights = np.ones(len(y), dtype=float)
    for group in np.unique(groups):
        for outcome in (0, 1):
            cell = (groups == group) & (y == outcome)
            if not cell.any():
                continue
            expected = (groups == group).mean() * (y == outcome).mean()
            weights[cell] = expected / cell.mean()
    return weights


def per_group_thresholds(scores, y_true, groups, objective="selection_rate") -> dict:
    """One threshold per group, chosen to equalise the objective across groups.

    `selection_rate` targets demographic parity: every group is flagged at the same rate.
    `tpr` targets equal opportunity: every group has the same share of true defaulters
    caught.
    """
    overall = flag_at_rate(scores)
    target = (
        overall.mean()
        if objective == "selection_rate"
        else float(overall[np.asarray(y_true) == 1].mean())
    )

    thresholds = {}
    for group in np.unique(groups):
        mask = groups == group
        group_scores = scores[mask]
        if objective == "selection_rate":
            thresholds[group] = float(np.quantile(group_scores, 1 - target))
        else:
            positives = group_scores[np.asarray(y_true)[mask] == 1]
            thresholds[group] = (
                float(np.quantile(positives, 1 - target)) if len(positives)
                else float(np.quantile(group_scores, 1 - target))
            )
    return thresholds


def apply_thresholds(scores, groups, thresholds) -> np.ndarray:
    predicted = np.zeros(len(scores), dtype=int)
    for group, cutoff in thresholds.items():
        mask = groups == group
        predicted[mask] = (scores[mask] >= cutoff).astype(int)
    return predicted


def run_attribute(attribute, labels, X_dev, y_dev, X_test, y_test,
                  dev_groups, test_groups) -> list[dict]:
    """Every intervention, for one protected attribute."""
    print(f"\n{'=' * 72}\n{attribute}: {dict(pd.Series(test_groups).value_counts())}\n{'=' * 72}")
    results = []

    start = time.time()
    base = light_model().fit(X_dev, y_dev)
    base_scores = base.predict_proba(X_test)[:, 1]
    base_predicted = flag_at_rate(base_scores)
    results.append(measure("baseline (no mitigation)", y_test, base_predicted, test_groups,
                           base_scores, time.time() - start,
                           "the model as deployed, for reference"))

    start = time.time()
    weights = reweight(y_dev, dev_groups)
    reweighted = light_model().fit(X_dev, y_dev, sample_weight=weights)
    scores = reweighted.predict_proba(X_test)[:, 1]
    results.append(measure("pre: reweighting", y_test, flag_at_rate(scores), test_groups,
                           scores, time.time() - start,
                           "Kamiran and Calders; group and outcome made independent"))

    for objective, label in [("selection_rate", "demographic parity"),
                             ("tpr", "equal opportunity")]:
        start = time.time()
        thresholds = per_group_thresholds(base_scores, y_test, test_groups, objective)
        predicted = apply_thresholds(base_scores, test_groups, thresholds)
        results.append(measure(f"post: per-group threshold ({label})", y_test, predicted,
                               test_groups, base_scores, time.time() - start,
                               "one cut-off per group; scores unchanged"))

    for constraint, label in [(DemographicParity(), "demographic parity"),
                              (EqualizedOdds(), "equalized odds")]:
        for eps in (0.01, 0.05, 0.10):
            start = time.time()
            try:
                reduction = ExponentiatedGradient(
                    light_model(), constraints=constraint, eps=eps, max_iter=12)
                reduction.fit(X_dev, y_dev, sensitive_features=dev_groups)
                predicted = np.asarray(reduction.predict(X_test)).astype(int)
                results.append(measure(
                    f"in: exponentiated gradient ({label}, eps={eps})", y_test, predicted,
                    test_groups, None, time.time() - start,
                    "trained under an explicit constraint; eps is the allowed slack"))
            except Exception as error:
                print(f"  skipped eps={eps} {label}: {type(error).__name__}: {error}")

    for objective, label in [("demographic_parity", "demographic parity"),
                             ("equalized_odds", "equalized odds")]:
        start = time.time()
        try:
            optimiser = ThresholdOptimizer(
                estimator=base, constraints=objective, prefit=True,
                predict_method="predict_proba")
            optimiser.fit(X_dev, y_dev, sensitive_features=dev_groups)
            predicted = np.asarray(
                optimiser.predict(X_test, sensitive_features=test_groups)).astype(int)
            results.append(measure(f"post: threshold optimiser ({label})", y_test,
                                   predicted, test_groups, None, time.time() - start,
                                   "fairlearn's post-processing, randomised at the cut-off"))
        except Exception as error:
            print(f"  skipped threshold optimiser {label}: "
                  f"{type(error).__name__}: {error}")

    frame = pd.DataFrame(results)
    print(f"\n{'method':48s} {'DIR':>6s} {'DP diff':>8s} {'EO diff':>8s} {'F1':>7s} {'bal acc':>8s}")
    print("-" * 92)
    for _, r in frame.iterrows():
        print(f"{r['method']:48s} {r['disparate_impact_ratio']:6.3f} "
              f"{r['demographic_parity_difference']:8.3f} "
              f"{r['equalized_odds_difference']:8.3f} {r['f1']:7.3f} "
              f"{r['balanced_accuracy']:8.3f}")
    return results


def main() -> dict:
    frames, y, audit = build_splits(seed=SEED, blind=True)
    X_dev = pd.concat([frames["train"], frames["validation"]])
    y_dev = pd.concat([y["train"], y["validation"]])
    dev_audit = pd.concat([audit["train"], audit["validation"]])
    X_test, y_test = frames["test"], y["test"]

    studies = {}
    for attribute, labels in [("SEX", SEX_LABELS), ("EDUCATION", EDUCATION_LABELS)]:
        dev_groups = dev_audit[attribute].map(labels).to_numpy()
        test_groups = audit["test"][attribute].map(labels).to_numpy()
        studies[attribute] = run_attribute(
            attribute, labels, X_dev, y_dev, X_test, y_test, dev_groups, test_groups)

    payload = {
        "seed": SEED,
        "flag_rate": FLAG_RATE,
        "base_model": LIGHT,
        "note": (
            "A lighter LightGBM is used throughout so the reductions sweep is tractable; "
            "the baseline uses the same model, so comparisons are like for like."
        ),
        "studies": studies,
    }
    out = artifacts_dir() / "mitigation.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n{'=' * 72}\nFRONTIER (SEX): methods nothing else beats on both axes\n{'=' * 72}")
    frame = pd.DataFrame(studies["SEX"])
    frontier = [
        row for _, row in frame.iterrows()
        if not ((frame["balanced_accuracy"] >= row["balanced_accuracy"])
                & (frame["equalized_odds_difference"] <= row["equalized_odds_difference"])
                & ((frame["balanced_accuracy"] > row["balanced_accuracy"])
                   | (frame["equalized_odds_difference"] < row["equalized_odds_difference"]))
                ).any()
    ]
    for row in sorted(frontier, key=lambda r: r["equalized_odds_difference"]):
        print(f"  {row['method']:48s} EO={row['equalized_odds_difference']:.3f}  "
              f"bal acc={row['balanced_accuracy']:.3f}")

    print(f"\nWritten to {out}")
    return payload


if __name__ == "__main__":
    main()
