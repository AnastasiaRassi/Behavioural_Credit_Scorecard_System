# Credit Risk Modelling System: finished product and status

TM471 Graduation Project, FYP II. Arab Open University, Lebanon Branch.
Author: Anastasia Al Rassi · Supervisor: Dr. Maya Dawoud

**What this document is.** The definition of the finished product, what is already
built, and what remains. Update the status table as sections complete.

Last updated: 2026-11-04

---

## 1. What the finished product is

### Scope: three objectives plus fairness

Taken from proposal #15, *Explainable and Fair Credit Decision System* (one semester,
difficulty 3). Everything built serves one of these:

1. **Train accurate credit-risk models.**
2. **Explain decisions globally and per applicant** — SHAP, LIME, and counterfactuals
   (what would need to change for this customer to be approved).
3. **Measure and mitigate bias**, across sex, age, education and marital status.

Supporting these: customer segmentation by behavioural archetype, used to test whether
explanations stay consistent for similar customers.

**Research angle:** compare explanation methods for consistency.

**Out of scope**, each considered and dropped: performance decay *over time* and the
weekly replay and retraining policies that depend on it (the dataset is a single snapshot
with no time axis — `ml/drift.py` measures population stability between two cohorts, which
needs no time axis and is not the same claim); federated learning (needs multiple data holders, so one
bank's data would have to be split into institutions that never existed); the user study
on explanation clarity (ethics approval and recruitment; within-segment coherence tests
consistency without human subjects).

Note the one scope change against the proposal: the user study is listed there under
evaluation metrics, not deliverables, so both stated deliverables remain intact. The
research angle narrows from "consistency and user trust" to **consistency**, because
trust cannot be measured without participants.

### The system

An offline prototype credit risk platform with three roles. It scores credit card
customers for risk of missing their next payment, explains each score in plain language,
and audits itself for fairness.

**It is a behavioural scorecard, not an application scorecard.** The customer already
holds the product, and the model re-scores them as new repayment history arrives. Stating
this matters, because the model legitimately sees whether payments have already been
missed.

### The three roles

| Role | Sees | Why it exists |
|---|---|---|
| **Loan officer** | Customer input form, risk band, probability, plain language reasons, approve / decline / refer with a justification box, CSV batch upload | Fast, consistent, explainable decisions with a human in the loop |
| **ML engineer / admin** | Full metric report with explanations, fairness audit, feature importance, interactive threshold, model registry | Nothing ships without an audit; retrain on evidence |
| **Cardholder** | Their own account: whether the next payment looks at risk, the reasons in plain language, and what would help most | Right to explanation |

Full screen by screen design for the ML engineer view is in
[frontend_ml_engineer_view.md](frontend_ml_engineer_view.md). Code style and the artifact
contract are in [CODE_CONVENTIONS.md](CODE_CONVENTIONS.md).

### Definition of done

The project is finished when:

1. A loan officer can enter a customer, get a risk band, and see why.
2. An ML engineer can see every metric with a plain language explanation beside it, plus
   the fairness audit.
3. Both run against the same saved model through the same scoring code.
4. The report has a Results chapter written from these numbers.

---

## 2. Verified results

Measured once on a held out test set of 4,500 customers the model never saw. 22.1%
of them defaulted.

**Provenance.** Every figure below is read from `backend/artifacts/scoring_config.json`,
written by `python -m app.ml.train`. Model version `lightgbm_ensemble_v1`, trained
2026-10-11, 46 features, ensemble seeds (42, 202, 777), splits 21,000 / 4,500 / 4,500.
Nothing here is typed from memory; if the table and the artifact ever disagree, the
artifact is right and this section is stale.

Reported in four groups. The first four rows are what the notebook and the presentation
lead with; the rest are computed and stored for the paper.

| Metric | Value | Role |
|---|---|---|
| ROC-AUC | 0.783 | ranking quality, the selection metric |
| Gini | 0.566 | same information, banking convention |
| KS | 43.3 | separation, 58% of defaults against 15% of payers |
| F1, default class | 0.541 | operating point, with precision 0.478 and recall 0.622 |
| Brier | 0.134 | are the probabilities honest |
| PR-AUC | 0.558 | minority class view, baseline 0.221 |
| Balanced accuracy | 0.714 | both classes weighted equally |
| Specificity | 0.807 | good payers correctly left alone |
| Weighted F1 | 0.776 | quoted in papers, inflated by the majority class |
| Macro F1 | 0.692 | both classes counted equally |
| MCC | 0.393 | correlation with the truth |
| Cohen kappa | 0.387 | agreement beyond chance |

Confusion matrix at threshold 0.2389: tn 2826, fp 678, fn 376, tp 620.

### Risk bands

Edges chosen on validation: Low below 0.1471, Medium below 0.3312, High above it.

| Band | Customers | Actual default rate |
|---|---|---|
| Low | 2,224 | 8.9% |
| Medium | 1,338 | 20.3% |
| High | 938 | 56.3% |

### Context for the numbers

| Comparison | F1 |
|---|---|
| Coin flip | 0.306 |
| Flag everyone | 0.362 |
| Naive "was late last month" | 0.505 |
| **This model** | **0.541** |

The first two follow from the 22.1% test default rate alone. The naive rule is
`PAY_1 > 0` on the same 4,500 test rows: it flags 1,032 customers at precision 0.496 and
recall 0.514. The model is worth about 0.036 F1 over the rule a bank could apply with no
model at all, which is the honest size of the contribution.

AUC is stable at 0.78 plus or minus 0.01 across five independent data splits, so any
difference smaller than about 0.02 is noise.

---

## 3. Status

State is judged against what is in the repo, not against what was planned.

**Objective 1, accurate models**

| Section | State |
|---|---|
| Dataset selection and justification | ✅ Done |
| Data preparation and feature engineering | ✅ Done, 46 features, `ml/feature_engineering.py` |
| Model comparison, 9 families | ✅ Done, `ml/experiments.py` |
| Hyperparameter tuning, 70 configurations | ✅ Done, winner cached in `train.py` |
| Calibration and threshold selection | ✅ Done, isotonic, `ml/calibration.py` |
| Resampling investigation | ✅ Done, no benefit found, 5 strategies |
| Feature importance | ✅ Done, permutation, all 46 |
| Extra-feature studies | ✅ Done, `ml/feature_experiments.py` |
| Sequence models (neural baseline) | ✅ Done, `ml/sequence_models.py` |
| Training script producing all artifacts | ✅ Done, `ml/train.py` |
| Scoring core | ✅ Done, `CreditScorer` |

**Objective 2, explanations**

| Section | State |
|---|---|
| SHAP explanations | ✅ Done, native LightGBM `pred_contrib`, no `shap` dependency |
| LIME | ✅ Done, `ml/explanation_methods.py` |
| Occlusion | ✅ Done, third method for the comparison |
| Explanation stability comparison (research angle) | ✅ Done in code, `run_to_run_stability()` and `jaccard()` |
| Clustering into behavioural segments | ✅ Done, k=3 by silhouette (0.284): moderate user, distressed, dormant |
| Clustering algorithm comparison | ✅ Done, `ml/clustering_comparison.py` |
| Predictive multiplicity | ✅ Done, `ml/multiplicity.py`, notebook 05 |
| **Counterfactuals** | ⬜ **Not started — the one real gap** |

**Objective 3, fairness**

| Section | State |
|---|---|
| Fairness audit | ✅ Done, 4 attributes pass four-fifths (DIR 0.880 to 0.990) |
| Bias mitigation | ✅ Done, `ml/mitigation.py`, 5 interventions across pre/in/post |

**System**

| Section | State |
|---|---|
| FastAPI service | ✅ Done, 9 routers, `app/main.py` |
| Database schema, ingestion and seed | ✅ Done, `db/models/`, `db/ingest.py`, `db/seed.py` |
| Role layer | ✅ Done, 3 roles, deliberately not authentication |
| Loan officer screen | ✅ Done, `frontend/views/officer.py` |
| ML engineer dashboard | ✅ Done, 6 tabs: performance, fairness, explain, threshold, drift, registry |
| Cardholder screen | ✅ Done, `frontend/views/cardholder.py` |
| Model registry | ✅ Done, `frontend/tabs/registry.py` |
| Drift (population stability index) | ✅ Done, `ml/drift.py` |
| Test suite | ✅ Done, 6 files under `backend/tests/` |
| Code conventions | ✅ Done |
| ML engineer view design spec | ✅ Done |
| **Report chapters 4.4 and 7** | ⬜ Not started |

All three objectives are met. What remains is counterfactuals, the report, and
presentation work — no open research question blocks any of it.

---

## 4. What remains, in build order

The analysis and the system are built. What is left is one explanation method, the
writing, and making it all presentable.

> **The project is already complete and defensible. Everything below adds marks rather
> than rescuing them.**

**Counterfactuals.** The last piece of objective 2, and the only genuine gap. *Pay 15,000
more next month and you move to Medium risk* is the explanation a flagged customer can act
on, and it is the one thing the three existing methods cannot give: SHAP, LIME and
occlusion all explain the score backwards, none of them say what to do. Search only over
what the customer controls next month, flow the change through `engineer()` so the
engineered features stay consistent, and recompute through `CreditScorer` rather than
trusting the search's own estimate. "No single-month action reaches Low" is a valid result
and must be reported as one. One to two sittings.

**Report chapters 4.4 and 7.** Results and Appendices. Every number is already in
`backend/artifacts/`, so this is writing, not analysis. Can run alongside everything else.

**Presentation pass.** Hardening before polish: every screen must render something
meaningful when the API is down or an artifact is missing, and every figure on screen must
be checked against the artifact that produced it. Then legibility at projector resolution.

**Optional, if time allows.** A plain-language coaching layer over the counterfactual
output, for the cardholder screen. Worth doing only on the rule that the text layer
re-words facts and never produces them: the probability, the reasons and the target all
come from code, and a template path must be good enough to demo with no network. See
[PRESENTATION_PROMPTS.md](PRESENTATION_PROMPTS.md).

---

## 5. Decisions already made

**Dataset: UCI Default of Credit Card Clients.** 30,000 customers, 23 columns, no missing
values. Chosen over Bondora peer to peer lending and Home Credit Model Stability. Bondora
work is kept as preliminary material in notebooks 01 and 02. Home Credit was specified but
not started, and was shelved because its contribution depended on fairness drift over
time, which is out of scope.

**Frontend: Streamlit, not React.** `frontend/package.json` is an empty leftover from the
original React plan and can be deleted. Streamlit is Python only, with no npm and no build
step, so the whole project stays in one language. The 💭 explanation icons the dashboard
needs are native through `st.metric(help=...)` rather than a component to build. For one
person working alongside a job this saves weeks and an examiner sees the same screens.

**Single calibrated model, no rank blending.** The notebook's four model blend scored
marginally worse on test and its scores were not probabilities, so its threshold could not
be applied to a calibrated output. The model that is evaluated is the model that is served.

**Protected attributes excluded from inputs.** Sex, age, education and marital status are
kept only for auditing. Excluding them costs 0.0007 AUC, so there is no accuracy argument
for keeping them.

**Headline metrics: ROC-AUC and Gini, then KS, then the operating point. Never accuracy.**
Predicting that nobody defaults scores 77.9% accuracy on this data, so accuracy says more
about the class balance than about the model. Weighted F1 is no longer a headline figure for
the same reason: 78% of customers never default, so the easy class carries it.

**Metric reporting is split between the notebook and the paper.** The notebook shows only the
headline set, so the story stays readable: ROC-AUC and Gini for ranking, KS for separation,
precision, recall and F1 at the deployed threshold for the operating point, and Brier with the
decile table for calibration. The full thirteen metrics are still computed and written to
`scoring_config.json` on every run, and the FYP paper carries all of them with their
interpretations. Nothing needs recomputing to write that section, it is already in the
artifact.

**KS is quoted out of 100, not as a decimal.** 43.3 rather than 0.433, since that is how credit
teams write it. Above 40 is strong for a behavioural scorecard and above 60 would suggest the
target has leaked into the features.

**No resampling.** Five strategies were tested. None improved AUC, all degraded
calibration, and SMOTE was the worst. Threshold tuning is the correct lever and is already
in place.

---

## 6. Honesty register

Things a reader should not be misled about. Keep this section current.

- **The notebook's tuning section is out of date.** It still contains the hand picked
  placeholder parameters and claims they came from a cross validation search. The real
  search has since completed and the true parameters are in `backend/app/ml/train.py`.
  The notebook needs regenerating to match.
- **Tuning changed nothing.** The cross validated best configuration scored test AUC
  0.7811 against 0.7813 for hand picked defaults. Reported as evidence of the ceiling,
  not hidden.
- **Equalized odds gaps are not reliable.** Disparate impact is stable across splits and
  all four attributes pass four-fifths: marriage 0.990, sex 0.938, age 0.938, education
  0.880 (`fairness_detail.json`). The TPR, FPR and AUC spread columns move substantially
  between splits and sometimes reverse sign, so they must not be used to claim bias
  against a specific group. The apparent 4.4-point sex AUC gap is the clearest example: it
  did not replicate across five splits and the education gap reversed sign. Methodology
  material, not a finding.
- **0.78 is the dataset ceiling.** 70 configurations spanned 0.7865 to 0.7915 CV AUC.
  Plain logistic regression reaches 0.760. All sophistication adds about 0.02.
- **Higher published figures on this dataset are usually one of two artifacts.** Either
  accuracy quoted as if it were AUC, or SMOTE applied before the split. The second was
  reproduced here and inflated test AUC from 0.780 to 0.924.
- **`backend/artifacts/model_lightgbm.joblib` and `model_metadata.json` are stale**, left
  over from the notebook and describing a different champion model and threshold. The live
  model is `model_ensemble.joblib` with `scoring_config.json`. Both old files can be
  deleted, and per conventions section 16 a stale artifact is a reproducibility bug rather
  than clutter.
- **Section 2 of this document was wrong until 2026-11-04.** It quoted ROC-AUC 0.781,
  Gini 0.562, PR-AUC 0.540 and threshold 0.2747 from an earlier training run, while
  `scoring_config.json` held 0.783, 0.566, 0.558 and 0.2389 for the model actually served.
  The band counts and confusion matrix were off by the same drift. Section 2 is now read
  from the artifact and carries a provenance note. The lesson is the one conventions
  section 16 already states: a number that is typed rather than read goes stale silently.
- **`backend/artifacts/calibrator_isotonic.joblib` is stale too.** No calibrator is fitted
  any more — `ml/calibration.py` records that isotonic was measured and was worse on
  Brier, log loss and calibration error, collapsed 4,500 customers onto 66 distinct values
  and scored 51 of them as certainties. The file is a leftover from before that
  measurement and is not in `train.py`'s `WRITES`, so the next training run deletes it.
  Nothing loads it.
- **A documented command does not exist.** Section 7 below and CODE_CONVENTIONS section 16
  both name `python -m app.ml.dashboard_artifacts`. There is no such module. Everything the
  dashboard reads is written by `app.ml.train` and the individual study modules.
- **CODE_CONVENTIONS section 14 is out of date.** It says there is no test suite and that
  smoke assertions are current practice. `backend/tests/` now holds six files driven by
  FastAPI's TestClient.

---

## 7. Running it

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"
```

Place the dataset in `data/raw/`. Download from
<https://archive.ics.uci.edu/dataset/350/>. Data is gitignored and never committed.

Regenerate all model artifacts:

```bash
cd backend
python -m app.ml.train
```

That writes exactly the eight files in `train.py`'s `WRITES` manifest —
`model_ensemble.joblib`, `scoring_config.json`, `threshold_sweep.json`,
`feature_importance.json`, `cost_curve.json`, `fairness_detail.json`, `registry.json`,
`officer_queue.json` — and deletes anything else it finds, except the two segmentation
files it is told to preserve. No calibrator is fitted: the model's own probabilities are
better calibrated than isotonic or Platt on top of them, which `ml/calibration.py`
documents with the measurements.

The study modules write their own artifacts and are run the same way, each only when its
screen or report section needs refreshing:

```bash
python -m app.ml.experiments            # experiment_results.json
python -m app.ml.feature_experiments    # feature_experiments.json
python -m app.ml.segmentation           # segments.json, segmenter.joblib
python -m app.ml.clustering_comparison  # clustering_comparison.json
python -m app.ml.explanation_methods    # explanation stability
python -m app.ml.mitigation             # the fairness-cost frontier
python -m app.ml.multiplicity           # multiplicity.json
python -m app.ml.sequence_models        # sequence_models.json
```

Run the system, both from the repo root:

```bash
cd backend && uvicorn app.main:app --port 8014
streamlit run frontend/app.py
```

Tests:

```bash
cd backend && python -m pytest
```

Open the notebooks `01_data_preparation` through `05_predictive_multiplicity` with the
**Python (credit-risk)** kernel for the full analysis.
