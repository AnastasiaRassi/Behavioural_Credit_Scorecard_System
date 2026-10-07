# Credit Risk Modelling System

An intelligent, data-driven credit-risk assessment platform. The system scores borrower
default risk, audits algorithmic fairness across demographic groups, and produces local
decision explanations.

Final Year Project (TM471), Arab Open University Lebanon.

---

## Scope

Three objectives, plus fairness. Everything in this repository serves one of them.

1. **Train accurate credit-risk models.**
2. **Explain decisions globally and per applicant** — SHAP, LIME, and counterfactuals
   (what would need to change for this customer to be approved).
3. **Measure and mitigate bias**, across sex, age, education and marital status.

Supporting these: customer segmentation by behavioural archetype, used to test whether
explanations stay consistent for similar customers.

**Research angle:** compare explanation methods for consistency. Tree SHAP is exact and
deterministic; LIME samples randomly and can return a different explanation for the same
customer on repeated runs. That difference is measurable, and it matters for any system
claiming to support a right to explanation.

### Deliberately out of scope

One semester, one person. These were each considered and dropped, with reasons:

| Excluded | Why |
|---|---|
| Performance decay and drift monitoring over time | The dataset is a single six-month snapshot with no time axis, so it cannot be measured |
| Weekly replay and retraining policies | Same reason; depends on the monitoring above |
| Federated learning | Needs multiple data holders. One bank's data would have to be split into institutions that never existed, making the result an artefact of the simulation |
| User study on explanation clarity | Needs ethics approval and participant recruitment. Within-segment explanation coherence tests consistency without human subjects |

Anything not serving an objective above does not get built, however interesting.

## What the model predicts

The current model is a **behavioural scorecard**: given a credit-card customer's last six
months of billing and repayment behaviour, it estimates the probability they will **default
on their next payment**.

This is distinct from an *application* scorecard, which decides whether to lend to a new
applicant using only their application form. The distinction matters when reading the
results — a behavioural model can see whether a customer has already begun missing payments,
which is legitimately available in this use case and is why behavioural models outperform
application models.

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
The dataset is **not** committed — `data/` is gitignored because it holds files above
GitHub's 100MB limit.

## Results

Evaluated once on a held-out test split of 4,500 customers that informed no modelling
decision.

| Metric | Test |
| --- | --- |
| ROC-AUC | **0.7807** |
| Gini | 0.5615 |
| KS | 0.4306 |
| PR-AUC | 0.5553 |
| F1 | 0.5364 |
| Precision | 0.4835 |
| Recall | 0.6024 |

Validation ROC-AUC was 0.7814 against a test figure of 0.7807, so the model generalises
without meaningful overfitting.

**Risk ranking.** Customers in the lowest predicted-risk decile default at 4.4%; the highest
decile defaults at 68.7% — a **15.5× lift**.

**Baseline comparison.** A naive rule that simply flags anyone late last month achieves
F1 0.510. The model reaches 0.536, trading a little precision for 8.5 points of recall. The
larger gain is qualitative: the naive rule yields one fixed binary flag, while the model
yields a calibrated probability at any chosen operating point, which is what makes risk
banding, cost-based thresholds and per-customer explanations possible.

### Fairness audit

Disparate impact ratio, measured on the test split. The four-fifths rule treats values below
0.80 as evidence of adverse impact.

| Attribute | DIR | Verdict |
| --- | --- | --- |
| SEX | 0.937 | Pass |
| AGE | 0.945 | Pass |
| EDUCATION | 0.888 | Pass |
| MARRIAGE | 0.988 | Pass |

Protected attributes are excluded from the model's inputs and retained only as audit
metadata. Removing them costs **0.0007 AUC** — effectively nothing — so there is no accuracy
argument for retaining them.

All four attributes pass disparate impact, but equalized-odds analysis reveals per-group
accuracy gaps that disparate impact alone does not capture: AUC is 0.799 for women against
0.755 for men, and 0.793 for university-educated customers against 0.745 for those educated
to high-school level. These gaps are reported in full in the notebook.

### Where the signal comes from

Permutation importance concentrates almost entirely in recent delinquency:

| Feature | AUC drop when shuffled |
| --- | --- |
| `pay_max` — worst delinquency across six months | 0.0275 |
| `PAY_1` — current repayment status | 0.0234 |
| `remaining_credit` — limit minus balance | 0.0117 |
| all 43 others combined | ~0.026 |

In plain terms: *has the customer missed a payment, how badly, and how much headroom is left
on the card.* Because permutation importance understates correlated features, and `PAY_1`
through `PAY_6` are strongly correlated, the defensible claim is that the delinquency block
dominates — not that exactly three features matter.

## Repository layout

```text
credit_risk_system/
├── notebooks/
│   └── 03_taiwan_modelling.ipynb    # Behavioural scorecard - model development
├── backend/
│   ├── app/ml/
│   │   ├── feature_engineering.py   # Shared by the notebook and the API
│   │   ├── train.py                 # Regenerates all artifacts
│   │   └── scorer.py                # Scoring core: probability + risk band
│   └── artifacts/                   # Serialized model, calibrator, config (gitignored)
├── data/
│   └── raw/                         # Datasets, downloaded not committed (gitignored)
├── frontend/
├── requirements.txt
└── README.md
```

### Earlier work on dataset selection

Two other datasets were evaluated before settling on this one. Their notebooks have been
removed from the working tree to keep the repository focused, and remain in git history at
commit `b20efea`.

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

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"
```

Open `notebooks/03_taiwan_modelling.ipynb`, select the **Python (credit-risk)** kernel and run
all cells. The notebook writes the fitted model, calibrator and metrics to
`backend/artifacts/`.

## Method

- **Splits** — stratified 70/15/15. Hyperparameters are chosen by 5-fold cross-validation on
  the training split; the validation split selects between model families, calibrators and
  thresholds; the test split is evaluated once and informs nothing.
- **Models compared** — logistic regression, LDA, Gaussian naive Bayes, MLP, random forest,
  extra trees, HistGradientBoosting, LightGBM, XGBoost, plus seed-averaged and rank-blended
  ensembles.
- **Selection metric** — ROC-AUC, chosen because it is invariant to class prevalence.
- **Calibration** — Platt scaling and isotonic regression compared on validation, since the
  raw ensemble score is a rank, not a probability.
- **Threshold** — reported both at the F1 optimum and across a cost curve sweeping the
  false-negative to false-positive cost ratio from 1:1 to 10:1, because the two errors carry
  very different costs for a lender.

## References

- Yeh, I. C. and Lien, C. H. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473–2480.
- Lessmann, S., Baesens, B., Seow, H. V. and Thomas, L. C. (2015). Benchmarking
  state-of-the-art classification algorithms for credit scoring. *European Journal of
  Operational Research*, 247(1), 124–136.
