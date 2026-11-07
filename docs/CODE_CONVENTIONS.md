# Code conventions

How the existing code is written, so later work matches it. Derived from what is already
in `backend/app/ml/`, which is the only part currently implemented.

---

## 1. Layout and what goes where

```
backend/app/
├── ml/
│   ├── feature_engineering.py   # column constants, loading, cleaning, engineer(), splits
│   ├── train.py                 # fits the model, writes every artifact
│   ├── scorer.py                # CreditScorer: probability + risk band
│   ├── explainability.py        # Explainer: per customer reasons
│   ├── artifacts.py             # cached readers for what train.py wrote
│   ├── drift.py                 # population stability index
│   ├── fairness_audit.py        # empty; the audit is computed in train.py
│   └── preprocessing.py         # empty, redundant now
├── ai/
│   ├── coaching.py              # cardholder coaching text and the counterfactual payload behind it
│   ├── assistant.py             # Loana AI chat: fact sheet, guardrails, opening, template fallback
│   ├── llm_client.py            # OpenAI, else Ollama; returns an LlmCall, with no text on failure
│   └── local_llm.py             # Ollama client; returns an LlmCall, with no text on failure
├── api/
│   ├── dependencies.py          # scorer, explainer, session, live model version
│   ├── predictions.py           # score one, score a CSV, read a stored score
│   ├── decisions.py             # record and list officer interventions
│   ├── fairness.py              # the audit, read from artifacts
│   ├── admin.py                 # metrics, sweeps, importance, versions
│   └── monitoring.py            # drift and production performance
├── db/
│   ├── models/                  # the schema, one module per concern
│   │   ├── accounts.py          #   users, user_details: every person; details for customers
│   │   ├── customers.py         #   customer (account_number), monthly_statement, credit_outcome
│   │   ├── scoring.py           #   model_version, prediction
│   │   ├── decisions.py         #   decision, observed_outcome
│   │   ├── chat.py              #   chat_turn: every assistant reply, with how it was made
│   │   └── base.py              #   declarative base and shared defaults
│   ├── database.py              # engine, session_scope(), create_all()
│   ├── ingest.py                # load_taiwan(): the dataset into the tables
│   ├── statements.py            # the stored months pivoted back to a wide frame
│   ├── chat_log.py              # write a chat turn; summarise the log (GET /monitoring/assistant)
│   ├── customer_registry.py     # a customer from submitted columns; give_login; invented names
│   └── seed.py                  # demo accounts and the live model version
├── schemas/                     # pydantic request and response shapes
├── core/config.py               # DATABASE_URL, read once from .env
├── core/passwords.py            # scrypt hashing; the only plaintext handler
├── main.py                      # the FastAPI app and /health
└── artifacts/                   # generated, gitignored
```

`users` holds credentials, but storing a hash is not a login. How a request
proves who is making it is a separate concern from this schema; what the schema
guarantees is that `password_hash` is never readable. `core/passwords.py` hashes
with stdlib scrypt and nothing else touches a plaintext password, and the seed
stores a random one per account rather than a shared demo credential.

**Rule: every person is a `users` row; a `customer` is what is scored about one.**
Staff are users with no details. Each of the 30,000 dataset entities is a user with
role `cardholder`, a `user_details` row (the audited demographics) and a `customer`
(the limit, the history, the label), so `customer.user_id` is never null. Whether a
user can sign in is a property of the row: `email` and `password_hash` are both set
or both null, and only a cardholder may have neither. Registering is attaching a
login to a customer's existing user (`customer_registry.give_login`), never
creating a second person. Names are fictional and live only on
`users.display_name`; nothing duplicates them.

**Rule: ids are UUIDs, and the dataset's number is `customer.account_number`.**
`users.id`, `customer.id` and every FK to them are UUIDs, so a sequence never
reaches a URL. The source file's ID (or a walk-in's `W-...`) is kept as the unique
`account_number`: what an officer types and what the ML queue artifact is keyed
on. `ml/worklist.py` therefore speaks account numbers and the API adds the
customer's UUID to a row (`dependencies.queued_account`), so the join between the
two lives in one place. `coaching.json` is keyed by account number for the same
reason: it survives a database rebuild that mints new UUIDs.

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

`train.py` writes, everything else reads. Nothing else writes to `artifacts/`, with one
exception: `coaching.json`, written by `db/seed.py` (see the table below).

| File | Contents |
|---|---|
| `model_ensemble.joblib` | list of 3 LightGBM models, one per seed |
| `scoring_config.json` | threshold, band edges, 13 metrics, confusion matrix, feature names |
| `threshold_sweep.json` | 86 threshold points with precision, recall, F1, counts |
| `feature_importance.json` | permutation importance, all 46 features |
| `coaching.json` | pre-generated coaching for the demo cardholders, from `db/seed.py`. Kept by `train.py` (`PRESERVES`); ignored once its `model_version` differs from the live model |

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

**The assistant's LLM is chosen by `ai/llm_client.py`.** With `OPENAI_API_KEY` in `.env` it
is OpenAI `gpt-4o-mini` (the `openai` package, pinned), and data leaves the machine: the
fact sheet only, never the probability or a protected attribute. Without a key it is
Ollama, a separate local server (`ollama pull qwen2.5:3b`; `OLLAMA_URL` and `OLLAMA_MODEL`
override the defaults), reached over HTTP with `requests`, where nothing leaves the
machine. With neither, templates answer.

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
