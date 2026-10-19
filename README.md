# Behavioural Credit Scorecard System

An explainable, fair credit-risk assessment platform. The system scores a credit-card
customer's risk of missing their next payment, explains each score in plain language, audits
itself for fairness across demographic groups, and serves all of it through an API and a
three-role dashboard.

Final Year Project (TM471), Arab Open University Lebanon.

---

## Scope

Three objectives, plus fairness. Everything in this repository serves one of them.

1. **Train accurate credit-risk models.** Done.
2. **Explain decisions globally and per applicant.** Tree SHAP is in production; occlusion is
   implemented and used across six models in the multiplicity study; LIME is implemented but
   its stability comparison has not been run.
3. **Measure and mitigate bias**, across sex, age, education and marital status. Measurement
   is done and audited on every training run; five mitigation methods are implemented and not
   yet run.

Supporting these: behavioural segmentation, used to test whether the risk model and its
explanations hold up inside each behavioural archetype rather than only on average.

### Research angle

**How arbitrary is a model-based credit decision, and the reason the customer is given for
it?** A right to explanation assumes there is *an* explanation. If two models a bank could
equally defend decline different people and cite different reasons for the people they both
decline, then "the model said so" is a policy choice dressed as a technical one. The question
splits in two, and the harder half is answered:

- **Across models, with the explanation method held fixed — done.** Six configurations within
  0.0088 AUC of each other disagree about 8.5% of customers; 4.3% would have their decision
  reversed by swapping the deployed model for an equally accurate one. Reasons are worse than
  decisions: among customers every model flags, *no* customer receives the same top three
  drivers from all six, and the single top driver is unanimous for only 12.4%. The disagreement
  is 1.96× concentrated in the Distressed segment, the customers with the most at stake. All
  six are explained by the same model-agnostic occlusion method on purpose, so model
  disagreement is not confounded with method disagreement.
- **Across methods, with the model held fixed — implemented, not run.** Tree SHAP is exact and
  deterministic; LIME samples randomly and can return a different explanation for the same
  customer on a second call. `explanation_methods.py` puts all three behind one interface for
  this comparison.

Three further findings fall out of the modelling work and are reported rather than buried:
that this dataset has a hard **0.78 ceiling** no amount of method clears, and that published
figures above it are generally artifacts; that **calibrating a log-loss-trained booster made it
worse** on every measure, against the usual advice to always calibrate; and that **disparate
impact is stable across splits while equalized-odds gaps are not**, so per-group error gaps
cannot support claims about a specific group on a sample this size.

### Deliberately out of scope

One semester, one person. These were each considered and dropped, with reasons:

| Excluded | Why |
|---|---|
| Performance decay and drift monitoring over time | The dataset is a single six-month snapshot with no time axis, so it cannot be measured. The drift endpoint computes PSI against the training distribution, which detects input shift but cannot validate outcome decay |
| Weekly replay and retraining policies | Same reason; depends on the monitoring above |
| Federated learning | Needs several holders of the *same* features who cannot pool their data. A second dataset does not supply that: Bondora and Home Credit answer a different prediction task with a different schema, so there is no shared model to average. The alternative is splitting one bank's customers into institutions that never existed, which makes the result an artefact of the partition |
| User study on explanation clarity | Needs ethics approval and participant recruitment. Within-segment explanation coherence tests consistency without human subjects |

Anything not serving an objective above does not get built, however interesting.

## What the model predicts

The model is a **behavioural scorecard**: given a credit-card customer's last six months of
billing and repayment behaviour, it estimates the probability they will **default on their
next payment**.

This is distinct from an *application* scorecard, which decides whether to lend to a new
applicant from their application form alone. The distinction matters when reading the results:
a behavioural model can see whether a customer has already begun missing payments, which is
legitimately available in this use case and is why behavioural models outperform application
models.

The prediction is a genuine forecast, not a restatement of the inputs:

- **48.3%** of customers who default next month were *not* late in the current month.
- **49.7%** of customers who *are* late this month recover and do not default.

Roughly half of all defaults come from customers who currently look healthy, and half of the
customers who look troubled turn out fine.

## Dataset

**UCI Default of Credit Card Clients** (Yeh & Lien, 2009) — 30,000 Taiwanese credit-card
customers observed April–September 2005, with the October outcome as the target.

- 23 raw features, expanded to 46 through feature engineering
- 22.1% default rate
- No missing values
- Protected attributes available for fairness auditing: `SEX`, `AGE`, `EDUCATION`, `MARRIAGE`

Download from <https://archive.ics.uci.edu/dataset/350/> and place the file in `data/raw/`.
The dataset is **not** committed — `data/` is gitignored.

Splits are stratified 70/15/15 (21,000 / 4,500 / 4,500). The final model is refitted on train
plus validation, 25,500 rows, and evaluated once on the test split.

## Results

Served model: an ensemble of three LightGBM fits over seeds 42, 202 and 777, evaluated on a
held-out test split of 4,500 customers that informed no modelling decision. Operating
threshold 0.2389, the F1 optimum.

| Metric | Test |
| --- | --- |
| ROC-AUC | **0.7830** |
| Gini | 0.5661 |
| KS | 43.3 |
| PR-AUC | 0.5584 |
| Brier | 0.1342 |
| F1 | 0.5405 |
| Precision | 0.4777 |
| Recall | 0.6225 |
| Specificity | 0.8065 |
| Balanced accuracy | 0.7145 |
| MCC | 0.393 |

Confusion matrix at the deployed threshold: tn 2,826, fp 678, fn 376, tp 620.

Validation ROC-AUC is 0.7831 for the same configuration fitted on the training split alone,
against 0.7830 on test, so the model generalises without meaningful overfitting.

**Accuracy is never reported as a headline.** Predicting that nobody defaults scores 77.9% on
this data, which says more about the class balance than about any model.

**Risk ranking.** Customers in the lowest predicted-risk decile default at 4.0%; the highest
decile defaults at 69.1%, a **17.3× lift**. Risk bands behave the same way:

| Band | Customers | Actual default rate |
| --- | --- | --- |
| Low | 2,224 | 8.9% |
| Medium | 1,338 | 20.3% |
| High | 938 | 56.3% |

**Baseline comparison.** A naive rule that flags anyone late last month reaches F1 0.5049 on
the same test split; flagging everyone reaches 0.3624; the model reaches 0.5405. The larger
gain is qualitative: the naive rule yields one fixed binary flag, while the model yields a
calibrated probability at any chosen operating point, which is what makes risk banding,
cost-based thresholds and per-customer explanations possible.

**Calibration.** The model's own probabilities are published unchanged. LightGBM is fitted
with a log-loss objective, so its output is already a probability, and both candidate
calibrators measured worse: Brier 0.1342 and ECE 0.0088 raw, against 0.1349 / 0.0181 for
isotonic and 0.1348 / 0.0233 for Platt. Isotonic also collapsed 4,500 customers onto 66
distinct values and returned exactly 1.0 for the riskiest, so it was removed.
`app/ml/calibration.py` records the comparison.

### Fairness audit

Disparate impact ratio, measured on the test split. The four-fifths rule treats values below
0.80 as evidence of adverse impact.

| Attribute | DIR | Verdict | Per-group AUC spread |
| --- | --- | --- | --- |
| SEX | 0.938 | Pass | 0.044 |
| AGE | 0.938 | Pass | 0.024 |
| EDUCATION | 0.880 | Pass | 0.050 |
| MARRIAGE | 0.990 | Pass | 0.001 |

Protected attributes are excluded from the model's inputs and retained only as audit metadata.
Removing them costs **0.0007 AUC** — effectively nothing — so there is no accuracy argument
for retaining them. The literature calls this *fairness through unawareness* and notes that
correlated proxies survive it, which is why outcomes are audited by group rather than treating
the exclusion as proof of fairness.

All four attributes pass disparate impact, but per-group accuracy gaps remain that disparate
impact alone does not capture: AUC is 0.802 for women against 0.757 for men, and 0.795 for
university-educated customers against 0.744 for those educated to high-school level. Groups too
small to measure reliably are suppressed rather than reported with a misleading figure: age
60+ (35 customers), education "other" (68) and marriage "other" (54).

**Equalized-odds gaps are not reliable here.** Disparate impact is stable across splits, but
the TPR, FPR and AUC spread columns move substantially between splits and sometimes reverse
sign, so they must not be used to claim bias against a specific group.

### Where the signal comes from

Permutation importance against test AUC concentrates almost entirely in recent delinquency:

| Feature | AUC drop when shuffled |
| --- | --- |
| `PAY_1` — current repayment status | 0.0289 |
| `pay_max` — worst delinquency across six months | 0.0138 |
| `remaining_credit` — limit minus balance | 0.0066 |
| all 43 others | none above 0.0026 |

In plain terms: *has the customer missed a payment, how badly, and how much headroom is left
on the card.* Because permutation importance understates correlated features, and `PAY_1`
through `PAY_6` are strongly correlated, the defensible claim is that the delinquency block
dominates, not that exactly three features matter.

## Studies beyond the headline model

Each of these is a notebook plus a module that writes its results to `backend/artifacts/`, so
nothing in the report is recomputed by hand.

**Added features** (`01_data_preparation.ipynb` §4). Behavioural concepts an analyst would ask
for by name — the run of arrears still open, payment volatility, recent versus historical
behaviour, a relative utilisation slope, and counts of how often a bill was cleared or barely
touched — engineered alongside the served set as nine extra columns and then tested. Variants
that merely restated a served feature were dropped before testing rather than carried: a
longest-arrears run at 0.998 rank correlation with `pay_months_late`, an absolute utilisation
slope at 0.96 with `util_trend`, a utilisation shift at 0.91, and the payment-to-bill ratio,
which `repay_ratio_*` already computes by the same formula. What remains is distinct, the
closest being `months_cleared_bill` at 0.85, with `payment_shift_ratio` properly independent
at 0.30.

Several are strong alone: `delinquency_run_current` reaches 0.70 AUC by itself against 0.73 for
the served `pay_months_late`, and separates a 13.7% default rate at zero open months from 69.8%
at six. None of it pays. Together they move validation AUC from 0.7831 to 0.7822, the best
single group gains +0.0003 against a seed spread of 0.0003, and a five-split repeat through the
full deployed pipeline reaches the same conclusion on AUC, F1 and recall at a 20% alert rate.
They are kept as analysis and **not** promoted into the served feature set. Distinctness proved
necessary but not sufficient: a feature can be unlike anything present and still carry no
information the model lacks.

**Additional studies** (`04_additional_studies.ipynb`). Checks that feed no decision, where
notebook 02 holds the choices that build the served model. Sections 1 to 4 are scored on
validation so the test split stays reserved. Class balancing: seven strategies, none beat the
0.7829 baseline by more than noise (Tomek links 0.7833 is the best of them) and all degraded
calibration, with SMOTE at 0.7799 and ADASYN at 0.7777 the weakest. Ensembling: a single tuned
LightGBM at 0.7831 beat soft voting (0.7803 to 0.7820), stacking (0.7824) and hard voting
(0.6455). Model families: sklearn GradientBoosting 0.7807, ExtraTrees 0.7734,
HistGradientBoosting 0.7724, AdaBoost 0.7704, KNN 0.7691, linear SVM 0.7531. Deep learning: six
architectures, best 0.7798 from a four-layer PyTorch network, with sklearn MLPs between 0.7677
and 0.7747. Nothing beat the deployed configuration. Sections 5 and 6 add the sequence models
and the feature-side studies, both paired across five independent splits.

**Behavioural segmentation** (`03_unsupervised_modelling.ipynb`). K-means on an interpretable
subset of the engineered features, k = 3 chosen by silhouette (0.284). The archetypes are the
ones card issuers use:

| Segment | Customers | Share | Default rate |
| --- | --- | --- | --- |
| Moderate user | 2,757 | 61% | 11.7% |
| Distressed | 1,008 | 22% | 48.0% |
| Dormant | 735 | 16% | 25.7% |

The notebook also checks whether the segments are demographic proxies, whether the risk model
performs equally well inside each, and how the choice of clustering algorithm changes the
answer.

**Predictive multiplicity** (`05_predictive_multiplicity.ipynb`). Six models within 0.0088 AUC
of each other disagree about individuals: 8.5% of customers are flagged by some and cleared by
others, and 4.3% would have their decision reversed by swapping the deployed model for an
equally accurate one. Reasons diverge further than decisions. Among customers every model
flags, **no** customer receives the same top three drivers from all six models, the single top
driver is unanimous for only 12.4%, and mean pairwise overlap is 0.49. The ambiguity is not
spread evenly: it is 1.96× concentrated in the Distressed segment. Under a right to
explanation this makes the choice of model a policy decision rather than a technical one.

## The system

No authentication. The role selector opens a set of screens; it proves nothing, and the API
says the same. A real deployment would put an identity provider in front of both, which is
more honest than a login form that checks nothing.

| Role | Sees |
| --- | --- |
| **Loan officer** | Worklist, customer detail, risk band, probability, plain-language reasons, approve / decline / refer with a justification box, CSV batch upload |
| **ML engineer / admin** | Performance, fairness, feature importance, interactive threshold, drift, model registry |
| **Cardholder** | Own status and a simplified reason, reduced server-side so what a role cannot see is never sent |

FastAPI serves eight routers (`predictions`, `officer`, `cardholders`, `decisions`, `fairness`,
`monitoring`, `admin`, `roles`); Streamlit renders six admin tabs plus the officer and
cardholder views and reads nothing from disk. 94 tests cover the API, the role boundaries and
the database schema.

**The UI and API never recompute metrics.** `train.py` writes every artifact and everything
else reads them, which keeps the screens fast and guarantees what is displayed matches what was
actually evaluated.

## Repository layout

```text
credit_risk_system/
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Cleaning, feature engineering, splits, candidate features
│   ├── 02_supervised_modelling.ipynb   # Model comparison, tuning, threshold, fairness, importance
│   ├── 03_unsupervised_modelling.ipynb # Segmentation and clustering comparison
│   ├── 04_additional_studies.ipynb     # Resampling, ensembles, families, nets, sequences, features
│   └── 05_predictive_multiplicity.ipynb
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── feature_engineering.py  # Single source of truth, shared by notebooks and API
│   │   │   ├── train.py                # Writes every artifact
│   │   │   ├── scorer.py               # Probability and risk band
│   │   │   ├── explainability.py       # Per-customer reasons, native LightGBM SHAP
│   │   │   ├── explanation_methods.py  # SHAP, LIME and occlusion behind one interface
│   │   │   ├── calibration.py          # Why nothing is fitted on top of the model
│   │   │   ├── segmentation.py         # Behavioural segments
│   │   │   ├── multiplicity.py         # Predictive multiplicity study
│   │   │   ├── feature_experiments.py  # Which inputs are worth including
│   │   │   ├── sequence_models.py      # Reading the months in order
│   │   │   ├── mitigation.py           # Five bias-mitigation methods
│   │   │   ├── experiments.py          # Comparison studies
│   │   │   ├── fairness_audit.py       # Audit (computed in train.py)
│   │   │   ├── drift.py                # Population stability index
│   │   │   └── worklist.py             # Officer queue, reduced per role
│   │   ├── api/                        # Routing only, no scoring maths
│   │   ├── db/                         # SQLAlchemy models, session, seed
│   │   ├── schemas/                    # Pydantic request and response shapes
│   │   └── main.py                     # The FastAPI app and /health
│   ├── artifacts/                      # Generated, gitignored
│   └── tests/                          # 94 tests
├── frontend/                           # Streamlit: tabs/ for admin, views/ for officer and cardholder
├── docs/
│   ├── PROJECT_STATUS.md               # Definition of done and status
│   ├── CODE_CONVENTIONS.md             # How the code is written, and the artifact contract
│   └── frontend_ml_engineer_view.md    # Screen-by-screen design spec
├── data/raw/                           # Downloaded, not committed
├── requirements.txt
└── README.md
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"
```

Place the dataset in `data/raw/`, then regenerate the model and every artifact the API and
dashboard read:

```bash
cd backend
python -m app.ml.train
```

The study artifacts are separate modules, each writing one file:

```bash
python -m app.ml.experiments            # experiment_results.json
python -m app.ml.segmentation           # segments.json, segmenter.joblib
python -m app.ml.clustering_comparison  # clustering_comparison.json
python -m app.ml.multiplicity           # multiplicity.json
```

Run the API, then the dashboard:

```bash
cd backend && uvicorn app.main:app --port 8014
streamlit run frontend/app.py
```

The notebooks run end to end on the **Python (credit-risk)** kernel, in numerical order.

## Method

- **Splits** — stratified 70/15/15. Hyperparameters come from a randomized search scored by
  5-fold cross-validation on the training split; the validation split selects between model
  families and operating points; the test split is evaluated once and informs nothing.
- **Models compared** — logistic regression, LDA, Gaussian naive Bayes, KNN, linear SVM, MLP
  and PyTorch networks, random forest, extra trees, AdaBoost, sklearn GradientBoosting,
  HistGradientBoosting, LightGBM, XGBoost, plus seed-averaged, voting and stacked ensembles.
- **Selection metric** — ROC-AUC, chosen because it is invariant to class prevalence.
- **Served model** — three LightGBM fits over fixed seeds, averaged. Seeds are explicit
  everywhere; the tuned parameters are cached in `train.py` with the search that produced them
  recorded, and `train.py` asserts they match what `scoring_config.json` claims.
- **Calibration** — none applied, having been measured as worse than the raw probabilities.
- **Threshold** — reported both at the F1 optimum and across a cost curve sweeping the
  false-negative to false-positive cost ratio from 1:1 to 10:1, because the two errors carry
  very different costs for a lender.
- **Reproducibility** — every number that reaches a screen or the report is regenerable from a
  clean `artifacts/` directory by the commands above. Anything that cannot be does not get
  published.

## Known gaps

Stated rather than hidden, since each one would otherwise mislead a reader.

- **No out-of-time sample.** The file carries no date to split on, so a random split cannot
  test whether the model holds in a later period. The window also sits at the onset of Taiwan's
  card-debt crisis, which is why the 22% default rate is high for the product.
- **0.78 is the dataset ceiling.** 70 tuning configurations spanned 0.7865 to 0.7915 CV AUC,
  plain logistic regression reaches 0.760, and the candidate-feature study adds nothing. All
  sophistication is worth roughly 0.02. Higher published figures on this dataset are usually
  either accuracy quoted as AUC or SMOTE applied before the split; the second was reproduced
  here and inflated test AUC from 0.780 to 0.924.
- **Objective 2 is incomplete.** The across-models half of the research angle is done; the
  across-methods half is not. `explanation_methods.py` implements SHAP, LIME and occlusion
  behind one interface, but run-to-run variance per method, agreement between methods and
  coherence within a segment have not been measured. Counterfactuals are not built.
- **Objective 3 has measurement but no reported mitigation.** `mitigation.py` implements
  reweighting, exponentiated gradient, per-group thresholds and fairlearn's threshold
  optimiser, and has not been run, so no mitigation frontier exists yet.
- **`02_supervised_modelling.ipynb` is out of sync with the served model.** Its tuning cell
  still holds earlier parameters and its persistence cell still fits an isotonic calibrator,
  both superseded by `train.py`. Treat `train.py` as authoritative until the notebook is
  regenerated.

### Earlier work on dataset selection

Two other datasets were evaluated before settling on this one. Their notebooks were removed
from the working tree to keep the repository focused, and remain in git history at commit
`b20efea`.

**Bondora P2P lending** — an application scorecard, predicting default within 12 months using
only information available at origination. It reached 0.711 test ROC-AUC against Bondora's own
published probability of default at 0.666, beating their internal model while using strictly
less information. A tiered experiment measured what relaxing the information boundary buys:
0.711 using application data alone, 0.729 adding origination pricing, 0.736 adding Bondora's
own credit rating. It was set aside because 33 sparse features and a 12-month outcome window
made iteration slow without improving on the behavioural task.

**Home Credit Model Stability** — designed for monitoring performance and fairness drift over
time. Not pursued; it answers a different research question and the multi-table data
engineering was out of proportion to the remaining schedule.

## References

- Yeh, I. C. and Lien, C. H. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473–2480.
- Lessmann, S., Baesens, B., Seow, H. V. and Thomas, L. C. (2015). Benchmarking
  state-of-the-art classification algorithms for credit scoring. *European Journal of
  Operational Research*, 247(1), 124–136.
