"""Comparison studies on the model side: resampling, ensembling, extra families, neural nets.

These are evidence for the report, not candidates for deployment. The served model is
whatever `train.py` builds; nothing here replaces it. Everything is scored on the
**validation** split so the test split stays reserved for the single final evaluation.

Four studies, each answering a question an examiner is likely to ask:

1. **Resampling** — does SMOTE or any other class-balancing technique help?
2. **Ensembling** — do voting and stacking beat a single tuned model?
3. **More families** — have we actually tried enough model types?
4. **Neural networks** — does deep learning beat gradient boosting on this data?

Run `python -m app.ml.experiments` to regenerate `experiment_results.json`.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from app.ml.feature_engineering import build_splits, project_root
from app.ml.train import HYPERPARAMETERS

SEED = 42
# Support vector machines are O(n^2) in the sample count; 21,000 rows would dominate the
# runtime of the whole study for a model that is not competitive here anyway.
SVM_SUBSAMPLE = 6000


def artifacts_dir() -> Path:
    path = project_root() / "backend" / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tuned_lgbm(seed: int = SEED) -> LGBMClassifier:
    """The deployed configuration, used as the reference every study compares against."""
    return LGBMClassifier(**HYPERPARAMETERS, n_jobs=-1, random_state=seed, verbose=-1)


def scaled(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def imputed(model):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def best_f1(y_true, scores) -> float:
    grid = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 120)))
    return max(f1_score(y_true, (scores >= t).astype(int)) for t in grid)


def evaluate(name, y_true, scores, seconds, note="") -> dict:
    predicted = (scores >= 0.5).astype(int)
    return {
        "name": name,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "best_f1": float(best_f1(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "seconds": round(seconds, 1),
        "note": note,
    }


def fit_score(name, model, X_fit, y_fit, X_eval, y_eval, note="") -> dict:
    start = time.time()
    model.fit(X_fit, y_fit)
    try:
        scores = model.predict_proba(X_eval)[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            # LinearSVC has no predict_proba; its decision function still ranks.
            scores = 1 / (1 + np.exp(-model.decision_function(X_eval)))
        except AttributeError:
            # Hard voting returns labels only. AUC on 0/1 is degraded by construction,
            # which is the point the comparison is making.
            scores = model.predict(X_eval).astype(float)
    result = evaluate(name, y_eval, scores, time.time() - start, note)
    print(f"  {name:34s} auc={result['roc_auc']:.4f}  f1={result['best_f1']:.4f}  "
          f"{result['seconds']:5.1f}s")
    return result


# --------------------------------------------------------------- study 1
def study_resampling(X_train, y_train, X_eval, y_eval) -> list[dict]:
    """Does class balancing help? Resamplers touch training data only, never evaluation."""
    print("\n[1/4] Resampling")
    imputer = SimpleImputer(strategy="median").fit(X_train)
    train_i = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
    eval_i = pd.DataFrame(imputer.transform(X_eval), columns=X_eval.columns)

    results = [
        fit_score("baseline (no resampling)", tuned_lgbm(), train_i, y_train, eval_i, y_eval,
                  "the deployed configuration")
    ]

    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    for label, model in [
        ("class_weight=balanced", LGBMClassifier(**HYPERPARAMETERS, class_weight="balanced",
                                                 n_jobs=-1, random_state=SEED, verbose=-1)),
        ("scale_pos_weight", LGBMClassifier(**HYPERPARAMETERS, scale_pos_weight=ratio,
                                            n_jobs=-1, random_state=SEED, verbose=-1)),
    ]:
        results.append(fit_score(label, model, train_i, y_train, eval_i, y_eval,
                                 "reweights the loss, does not resample"))

    for label, sampler in [
        ("SMOTE", SMOTE(random_state=SEED)),
        ("ADASYN", ADASYN(random_state=SEED)),
        ("random oversampling", RandomOverSampler(random_state=SEED)),
        ("random undersampling", RandomUnderSampler(random_state=SEED)),
        ("Tomek links cleaning", TomekLinks()),
    ]:
        X_res, y_res = sampler.fit_resample(train_i, y_train)
        results.append(
            fit_score(label, tuned_lgbm(), X_res, y_res, eval_i, y_eval,
                      f"train became {len(X_res):,} rows at {y_res.mean():.0%} default")
        )

    # The common mistake, quantified: resample everything, then split.
    combined = pd.concat([train_i, eval_i], ignore_index=True)
    combined_y = pd.concat([y_train, y_eval], ignore_index=True)
    X_leak, y_leak = SMOTE(random_state=SEED).fit_resample(combined, combined_y)
    from sklearn.model_selection import train_test_split

    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
        X_leak, y_leak, test_size=0.2, stratify=y_leak, random_state=SEED
    )
    results.append(
        fit_score("SMOTE BEFORE split (INVALID)", tuned_lgbm(), Xl_tr, yl_tr, Xl_te, yl_te,
                  "not comparable: synthetic rows appear in both train and test")
    )
    return results


# --------------------------------------------------------------- study 2
def study_ensembles(X_train, y_train, X_eval, y_eval) -> list[dict]:
    """Voting and stacking against the single tuned model."""
    print("\n[2/4] Ensembling")
    base = [
        ("lgbm", tuned_lgbm()),
        ("xgb", XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5,
                              min_child_weight=5, subsample=0.9, colsample_bytree=0.8,
                              tree_method="hist", eval_metric="auc", n_jobs=-1,
                              random_state=SEED)),
        ("rf", imputed(RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                              n_jobs=-1, random_state=SEED))),
        ("logreg", scaled(LogisticRegression(max_iter=3000))),
    ]

    results = [
        fit_score("single tuned LightGBM", tuned_lgbm(), X_train, y_train, X_eval, y_eval,
                  "reference")
    ]
    results.append(
        fit_score("soft voting (equal)", VotingClassifier(base, voting="soft", n_jobs=-1),
                  X_train, y_train, X_eval, y_eval, "averages predicted probabilities")
    )
    results.append(
        fit_score("soft voting (weighted 3,3,1,1)",
                  VotingClassifier(base, voting="soft", weights=[3, 3, 1, 1], n_jobs=-1),
                  X_train, y_train, X_eval, y_eval, "favours the boosted trees")
    )
    results.append(
        fit_score("soft voting (boosters only)",
                  VotingClassifier(base[:2], voting="soft", n_jobs=-1),
                  X_train, y_train, X_eval, y_eval, "lgbm + xgb")
    )
    results.append(
        fit_score("stacking (logreg meta)",
                  StackingClassifier(base, final_estimator=LogisticRegression(max_iter=2000),
                                     cv=3, n_jobs=-1),
                  X_train, y_train, X_eval, y_eval,
                  "meta-model learns how to combine, cross-fitted")
    )
    # Hard voting returns labels, not probabilities, so AUC is computed on a 0/1 vote share.
    results.append(
        fit_score("hard voting (majority)", VotingClassifier(base, voting="hard", n_jobs=-1),
                  X_train, y_train, X_eval, y_eval,
                  "labels not probabilities, so AUC is degraded by construction")
    )
    return results


# --------------------------------------------------------------- study 3
def study_more_families(X_train, y_train, X_eval, y_eval) -> list[dict]:
    """Model families not already covered, so 'did you try enough' has an answer."""
    print("\n[3/4] More model families")
    results = []
    for label, model, note in [
        ("AdaBoost", imputed(AdaBoostClassifier(n_estimators=300, learning_rate=0.5,
                                                random_state=SEED)), "boosted stumps"),
        ("GradientBoosting (sklearn)", imputed(GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)),
         "the original boosting implementation"),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=SEED), "sklearn's LightGBM equivalent"),
        ("ExtraTrees", imputed(ExtraTreesClassifier(n_estimators=600, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=SEED)),
         "randomised splits"),
        ("KNN (k=50)", scaled(KNeighborsClassifier(n_neighbors=50, n_jobs=-1)),
         "instance based, no training"),
    ]:
        results.append(fit_score(label, model, X_train, y_train, X_eval, y_eval, note))

    sample = X_train.sample(SVM_SUBSAMPLE, random_state=SEED)
    results.append(
        fit_score("Linear SVM (6k subsample)", scaled(LinearSVC(C=0.1, max_iter=5000)),
                  sample, y_train.loc[sample.index], X_eval, y_eval,
                  f"subsampled to {SVM_SUBSAMPLE:,}; SVMs scale quadratically")
    )
    return results


# --------------------------------------------------------------- study 4
class TorchTabularNet(BaseEstimator, ClassifierMixin):
    """A feed-forward network for tabular data, with the pieces scikit-learn's MLP lacks.

    Batch normalisation and dropout between layers, Adam, and early stopping on held-out
    AUC rather than training loss. Exposes `fit`/`predict_proba` so it drops into the same
    harness as every other model here.

    Expected to lose to gradient boosting: Grinsztajn et al. (2022) find trees dominate
    neural networks on tabular data, and reproducing that is the point of including it.
    """

    def __init__(self, layers=(256, 128, 64), dropout=0.3, epochs=120, batch_size=256,
                 learning_rate=1e-3, patience=12, seed=SEED):
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed

    def _build(self, n_features: int):
        import torch.nn as nn

        blocks, previous = [], n_features
        for width in self.layers:
            blocks += [
                nn.Linear(previous, width),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            previous = width
        blocks.append(nn.Linear(previous, 1))
        return nn.Sequential(*blocks)

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Carve a validation slice out of training for early stopping, so the study's
        # own evaluation split is never used to decide when to stop.
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(X))
        cut = int(len(X) * 0.85)
        fit_idx, stop_idx = order[:cut], order[cut:]

        self.model_ = self._build(X.shape[1])
        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X[fit_idx]), torch.from_numpy(y[fit_idx])),
            batch_size=self.batch_size, shuffle=True,
        )
        stop_X = torch.from_numpy(X[stop_idx])
        stop_y = y[stop_idx]

        best_auc, best_state, waited = -np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            for batch_X, batch_y in loader:
                optimiser.zero_grad()
                loss = loss_fn(self.model_(batch_X).squeeze(1), batch_y)
                loss.backward()
                optimiser.step()

            self.model_.eval()
            with torch.no_grad():
                held_out = torch.sigmoid(self.model_(stop_X).squeeze(1)).numpy()
            auc = roc_auc_score(stop_y, held_out)

            if auc > best_auc:
                best_auc, waited = auc, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_internal_auc_ = float(best_auc)
        # Pipeline introspects these; without them it cannot find predict_proba.
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(
                self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1)
            ).numpy()
        return np.column_stack([1 - p, p])


def study_neural(X_train, y_train, X_eval, y_eval) -> list[dict]:
    """Neural networks of increasing depth against gradient boosting."""
    print("\n[4/4] Neural networks")
    architectures = [
        ("MLP shallow (64)", (64,), "one hidden layer"),
        ("MLP (128, 64)", (128, 64), "two hidden layers"),
        ("MLP deep (256,128,64,32)", (256, 128, 64, 32), "four hidden layers"),
        ("MLP wide (512, 256)", (512, 256), "wide rather than deep"),
    ]
    results = []
    for label, layers, note in architectures:
        model = scaled(
            MLPClassifier(hidden_layer_sizes=layers, alpha=1e-3, max_iter=300,
                          early_stopping=True, n_iter_no_change=15, random_state=SEED)
        )
        results.append(fit_score(label, model, X_train, y_train, X_eval, y_eval, note))

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (PyTorch not installed; skipping the deep network)")
        return results

    for label, layers, dropout, note in [
        ("PyTorch (256,128,64)", (256, 128, 64), 0.3,
         "batch norm, dropout, Adam, early stopping on held-out AUC"),
        ("PyTorch deep (512,256,128,64)", (512, 256, 128, 64), 0.4,
         "deeper, more dropout"),
    ]:
        model = scaled(TorchTabularNet(layers=layers, dropout=dropout, seed=SEED))
        results.append(fit_score(label, model, X_train, y_train, X_eval, y_eval, note))
    return results


def main() -> dict:
    frames, y, _ = build_splits(seed=SEED, blind=True)
    X_train, y_train = frames["train"], y["train"]
    X_eval, y_eval = frames["validation"], y["validation"]

    print(f"train {len(X_train):,} rows, validation {len(X_eval):,} rows, "
          f"{y_train.mean():.1%} default")
    print("Scored on validation; the test split stays reserved.")

    studies = {
        "resampling": study_resampling(X_train, y_train, X_eval, y_eval),
        "ensembling": study_ensembles(X_train, y_train, X_eval, y_eval),
        "model_families": study_more_families(X_train, y_train, X_eval, y_eval),
        "neural_networks": study_neural(X_train, y_train, X_eval, y_eval),
    }

    payload = {
        "evaluated_on": "validation split",
        "rows": {"train": int(len(X_train)), "validation": int(len(X_eval))},
        "seed": SEED,
        "reference_hyperparameters": HYPERPARAMETERS,
        "studies": studies,
    }
    out = artifacts_dir() / "experiment_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 72)
    print("BEST PER STUDY (validation ROC-AUC)")
    print("=" * 72)
    for study, rows in studies.items():
        valid = [r for r in rows if "INVALID" not in r["name"]]
        best = max(valid, key=lambda r: r["roc_auc"])
        print(f"  {study:18s} {best['name']:34s} {best['roc_auc']:.4f}")
    print(f"\nWritten to {out}")
    return payload


if __name__ == "__main__":
    main()
