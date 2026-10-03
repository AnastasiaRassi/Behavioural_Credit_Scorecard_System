# Code conventions

How the existing code is written, so later work matches it. Derived from what is already
in `backend/app/ml/`, which is the only part currently implemented.

---

## 1. Layout and what goes where

```
backend/app/
├── ml/                          # implemented
│   ├── feature_engineering.py   # column constants, loading, cleaning, engineer(), splits
│   ├── train.py                 # fits the model, writes every artifact
│   ├── scorer.py                # CreditScorer: probability + risk band
│   ├── explainability.py        # Explainer: per customer reasons
│   ├── fairness_audit.py        # empty
│   ├── preprocessing.py         # empty, probably redundant now
│   └── drift.py                 # empty
├── api/                         # all empty
├── core/, db/, schemas/         # all empty
└── artifacts/                   # generated, gitignored
```

**Rule: `ml/` holds logic, `api/` holds only routing.** An endpoint should import from
`ml/` and translate between HTTP and Python. No scoring maths in a route handler.

## 2. The non negotiable one

**`feature_engineering.py` is the single source of truth.** Training and serving both
import `engineer()` from it. If a copy of that logic ever appears elsewhere, the served
model silently starts predicting on different features than it was trained on, and nothing
will error. The notebook imports from here too.

`engineer()` takes a frame with or without the target column, so the same function serves
training and live scoring:

```python
out = df.drop(columns=[TARGET], errors="ignore").copy()
```

Every feature uses only the customer's own row. No fitted population statistics, so there
is nothing to leak across splits.

## 3. The artifact contract

`train.py` writes, everything else reads. Nothing else writes to `artifacts/`.

| File | Contents |
|---|---|
| `model_ensemble.joblib` | list of 3 LightGBM models, one per seed |
| `calibrator_isotonic.joblib` | isotonic regression, raw probability to calibrated |
| `scoring_config.json` | threshold, band edges, 13 metrics, confusion matrix, feature names |
| `threshold_sweep.json` | 86 threshold points with precision, recall, F1, counts |
| `feature_importance.json` | permutation importance, all 46 features |

**The UI and API must never recompute metrics.** They read these files. That keeps the
screens fast and guarantees what is displayed matches what was actually evaluated.

To regenerate:

```bash
cd backend
python -m app.ml.train
```

## 4. Naming

- Modules and functions `snake_case`, classes `PascalCase`.
- Module level constants `UPPER_SNAKE`: `TARGET`, `PROTECTED`, `PAY_COLS`, `SEEDS`.
- Dataset column names keep their original casing exactly: `LIMIT_BAL`, `PAY_1`,
  `BILL_AMT3`. Engineered features are lowercase: `pay_max`, `util_mean`,
  `remaining_credit`.
- No abbreviations in new names. `calibrator` not `cal`, `probability` not `prob`.
- Private helpers take a leading underscore: `_prepare`, `_load`, `_money`.

## 5. Comments

Short. Explain the non obvious why, the constraint, or the gotcha. Never narrate what the
code plainly does, and never log change history.

```python
# Calibrator and operating points are fitted on validation, which the model has
# not seen. Fitting them on training predictions would make them over-confident.
```

```python
# Source dates are day-first (dd/mm/yyyy); pin it so parsing cannot silently
# flip to month-first.
```

Module docstrings state what the file is for and record any decision a reader would
otherwise question. `train.py`'s docstring explains why a single calibrated model is
served instead of the notebook's rank blend.

## 6. Typing and return shapes

Type hints on public functions. Dataclasses for structured returns, with a `to_dict()`
for the API layer:

```python
@dataclass
class ScoreResult:
    probability: float
    risk_band: str
    threshold: float
    flagged: bool
    band_default_rate: float

    def to_dict(self) -> dict:
        return asdict(self)
```

Pandas in, pandas out for batch work. Plain dicts at the API boundary.

## 7. Paths

Never rely on the working directory. Resolve from the module's own location:

```python
def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
```

`pathlib` throughout, never string concatenation.

## 8. Loading and caching

Artifacts load once per process. `@lru_cache` on the loader, so constructing a
`CreditScorer` is cheap and can happen per request:

```python
@lru_cache(maxsize=1)
def _load():
    ...
```

## 9. Errors

Fail loudly and early with a message that says what to do:

```python
raise ArtifactsMissing(
    f"Missing artifacts {missing} in {path}. Run: python -m app.ml.train"
)
```

Validate inputs before they reach the model. `scorer._prepare()` checks required columns
and that the engineered feature set matches what the model was trained on, because a
silent mismatch produces confident nonsense rather than an exception.

Custom exceptions over bare `RuntimeError` where a caller might want to catch it.

## 10. Protected attributes

Dropped inside `_prepare()`, not left to the caller to remember:

```python
features = features.drop(columns=PROTECTED, errors="ignore")
```

They are accepted in the input (the audit needs them) and removed before scoring. Any new
code path that reaches a model must go through `_prepare()`.

## 11. Randomness

Seeds explicit, never left to default. `SEEDS = (42, 202, 777)` for the ensemble, 42 for
splits. New random work takes a `seed` argument with a default, so results reproduce.
Reproducibility beyond seeds is §16.

## 12. User facing text

Explanation strings live in `explainability.py`'s `describe()`, not scattered through the
UI. Written for someone with no training:

- "No payment made last month" not "PAY_AMT1 = 0"
- "160,000 of credit still available" not "remaining_credit: 160000"

Numbers formatted at the point of display. Money with thousands separators, shares as
percentages.

## 13. Frontend

**Streamlit, not React.** `frontend/package.json` is an empty leftover and can be deleted.
Python only, no npm, no build step. The explanation icons the dashboard needs are native:

```python
st.metric("ROC-AUC", "0.781", help="Take one customer who defaulted and one who paid...")
```

Design is specified in [frontend_ml_engineer_view.md](frontend_ml_engineer_view.md).

## 14. Verification

No test suite yet. Current practice is a smoke check after each module, asserting against
a known value:

```python
# feature_engineering reproduces the notebook exactly
assert abs(auc - 0.7829) < 0.0005
```

Every module gained one before being considered done. If a real test suite arrives, these
become its first cases. `pytest` is already in `requirements.txt`.

## 15. Dependencies

Add to `requirements.txt` when adding an import. Currently added beyond the original list:
`xlrd` (reads the source `.xls`), `imbalanced-learn` (resampling experiment).

**`shap` is deliberately not a dependency.** LightGBM computes SHAP values natively with
`predict(X, pred_contrib=True)`, which for tree models is exact rather than sampled, and
avoids a large install. `explainability.py` uses that.

## 16. Reproducibility

Every number that reaches the report or a screen must be regenerable from a clean
`artifacts/` directory by documented commands:

```bash
cd backend
python -m app.ml.train
python -m app.ml.dashboard_artifacts
```

If a number cannot be produced that way it does not get published.

**Tuned constants carry their provenance.** `HYPERPARAMETERS` in `train.py` is the cached
output of a search, not a hand-picked set, and the comment above it names the search, the
number of configurations, the scoring scheme and the result. A tuned constant with no
recorded origin cannot be defended or re-derived, so the search itself stays in
version-controlled code with its own seed and grid — currently the tuning section of
`notebooks/modelling.ipynb`. Hardcoding the winner is acceptable, and keeps training fast;
hardcoding it without the search being re-runnable is not.

**Code and artifacts must agree.** `train.py`'s `HYPERPARAMETERS` and
`scoring_config.json`'s `hyperparameters` are the same values written twice, so they are
checked rather than trusted:

```python
# A silent divergence here means the published metrics describe a model the code
# no longer builds.
assert config["hyperparameters"] == HYPERPARAMETERS
```

**Artifacts describe what produced them.** `scoring_config.json` records hyperparameters,
seeds, feature names, split sizes and the training date, so a saved model can be audited
against the code that claims to build it without rerunning anything.

**Stale artifacts are a reproducibility bug, not clutter.** Anything in `artifacts/` not
written by the current training run is deleted. `model_lightgbm.joblib` and
`model_metadata.json` are notebook leftovers describing a different champion model and a
different threshold, and contradict the served model.

Dependencies are pinned in `requirements.txt` and every import appears there, so an
examiner installing the project gets the environment the numbers were produced in. See
also §11 on seeds and §15 on dependencies.
