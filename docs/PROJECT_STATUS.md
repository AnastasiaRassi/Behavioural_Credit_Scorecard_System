# Credit Risk Modelling System: finished product and status

TM471 Graduation Project, FYP II. Arab Open University, Lebanon Branch.
Author: Anastasia Al Rassi · Supervisor: Dr. Maya Dawoud

**What this document is.** The definition of the finished product, what is already
built, and what remains. Update the status table as sections complete.

Last updated: 2026-10-04

---

## 1. What the finished product is

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
| **Applicant** | Submit details, see status, read a simplified reason if declined, manage consent | Right to explanation |

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

Reported in four groups. The first four rows are what the notebook and the presentation
lead with; the rest are computed and stored for the paper.

| Metric | Value | Role |
|---|---|---|
| ROC-AUC | 0.781 | ranking quality, the selection metric |
| Gini | 0.562 | same information, banking convention |
| KS | 43.3 | separation, 58% of defaults against 15% of payers |
| F1, default class | 0.540 | operating point, with precision 0.477 and recall 0.623 |
| Brier | 0.135 | are the probabilities honest |
| PR-AUC | 0.540 | minority class view, baseline 0.221 |
| Balanced accuracy | 0.714 | both classes weighted equally |
| Specificity | 0.806 | good payers correctly left alone |
| Weighted F1 | 0.776 | quoted in papers, inflated by the majority class |
| Macro F1 | 0.692 | both classes counted equally |
| MCC | 0.393 | correlation with the truth |
| Cohen kappa | 0.387 | agreement beyond chance |

Confusion matrix at threshold 0.2747: tn 2825, fp 679, fn 376, tp 620.

### Risk bands

| Band | Customers | Actual default rate |
|---|---|---|
| Low | 2,231 | 8.9% |
| Medium | 1,348 | 20.4% |
| High | 921 | 56.7% |

### Context for the numbers

| Comparison | F1 |
|---|---|
| Coin flip | 0.306 |
| Flag everyone | 0.362 |
| Naive "was late last month" | 0.505 |
| **This model** | **0.540** |

AUC is stable at 0.78 plus or minus 0.01 across five independent data splits, so any
difference smaller than about 0.02 is noise.

---

## 3. Status

| Section | State |
|---|---|
| Dataset selection and justification | ✅ Done |
| Data preparation and feature engineering | ✅ Done, 46 features |
| Model comparison, 9 families | ✅ Done |
| Hyperparameter tuning, 70 configurations | ✅ Done |
| Calibration and threshold selection | ✅ Done |
| Fairness audit | ✅ Done, 4 attributes pass |
| Resampling investigation | ✅ Done, no benefit found |
| Feature importance | ✅ Done |
| Modelling notebook | ✅ Done and executed |
| Shared feature engineering module | ✅ Done |
| Training script producing all artifacts | ✅ Done |
| Scoring core | ✅ Done |
| ML engineer view design spec | ✅ Done |
| Code conventions | ✅ Done |
| **Section 2: SHAP explanations** | ✅ Done, native LightGBM, no `shap` needed |
| **Section 3: FastAPI service** | ⬜ Not started |
| **Section 4: Loan officer screen** | ⬜ Not started |
| **Section 5: ML engineer dashboard** | ⬜ Not started |
| **Section 6: Model registry** | ⬜ Not started, trimmable |
| **Section 7: Applicant role** | ⬜ Not started, demo level only |
| **Section 8: Report chapters 4.4 and 7** | ⬜ Not started |

Roughly a third complete, and it is the hardest third. Everything remaining is plumbing
rather than research. No open question blocks any of it.

---

## 4. What remains, in build order

**Section 3. FastAPI service.** Wrap the scoring core and the explainer. Endpoints for one
customer, a CSV batch, and the metric report the dashboard reads. Logic stays in `ml/`,
routes only translate HTTP. One to two sittings.

**Section 4. Loan officer screen.** Input form, risk band, reasons, decision buttons with
a justification box, CSV upload. Two sittings.

**Section 5. ML engineer dashboard.** Build to the existing spec. Performance, then
Fairness, then Threshold. Two to three sittings.

> **Stop here and the project is complete and defensible.**

**Section 6. Model registry.** Version list, promote and archive, fairness gate on
promotion. First thing to cut. Two sittings.

**Section 7. Applicant role.** Needs auth and a database, the most work for the least
credit. Build demo only, a role selector with no real login.

**Section 8. Report.** Chapter 4.4 Results and Chapter 7 Appendices. The analysis is
finished, so this is writing. Can run alongside everything else.

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
- **Equalized odds gaps are not reliable.** Disparate impact is stable across splits. The
  TPR, FPR and AUC spread columns move substantially between splits and sometimes reverse
  sign, so they must not be used to claim bias against a specific group.
- **0.78 is the dataset ceiling.** 70 configurations spanned 0.7865 to 0.7915 CV AUC.
  Plain logistic regression reaches 0.760. All sophistication adds about 0.02.
- **Higher published figures on this dataset are usually one of two artifacts.** Either
  accuracy quoted as if it were AUC, or SMOTE applied before the split. The second was
  reproduced here and inflated test AUC from 0.780 to 0.924.
- **`backend/artifacts/model_lightgbm.joblib` is stale**, left over from the notebook. The
  live model is `model_ensemble.joblib`. The old file can be deleted.

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

That writes `model_ensemble.joblib`, `calibrator_isotonic.joblib`,
`scoring_config.json`, `threshold_sweep.json` and `feature_importance.json`, which is
everything the API and dashboard read.

Open `notebooks/modelling.ipynb` with the **Python (credit-risk)** kernel for
the full analysis.
