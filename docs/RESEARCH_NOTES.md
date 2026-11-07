# Research notes

Scope, studies, method and known gaps for the Behavioural Scorecard System, moved out of the README. See the [README](../README.md) for the system itself.

---

## Scope and research angle

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
