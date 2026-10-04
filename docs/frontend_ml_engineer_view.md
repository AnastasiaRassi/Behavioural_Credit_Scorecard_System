# ML Engineer view: frontend specification

Design spec for the Model Operations dashboard. Every number shown here is real, taken
from the current model, so this doubles as the expected output once wired up.

**Principle.** Two levels of explanation on every metric. A one line caption always
visible underneath the number, and a 💭 icon that opens a longer plain language
explanation. Nobody should have to already know what a Brier score is.

**Second principle.** The dashboard says which numbers to trust. Some of our figures are
stable across data splits and some are not. The UI marks them rather than presenting
everything as equally solid.

---

## Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CREDIT RISK · MODEL OPERATIONS                          [ML Engineer ▾] │
├──────────────────────────────────────────────────────────────────────────┤
│  lightgbm_ensemble_v1   ·   trained 2026-10-03   ·   ● HEALTHY           │
│  25,500 training rows  ·  46 features  ·  seeds 42/202/777               │
├──────────────────────────────────────────────────────────────────────────┤
│  Performance │ Fairness │ Explainability │ Threshold │ Registry │ Drift  │
└──────────────────────────────────────────────────────────────────────────┘
```

Status light: green when every fairness check passes and test AUC is within 0.02 of
the recorded baseline. Amber on drift warning. Red on a fairness breach.

---

## Tab 1 · Performance

### Headline row

Four large tiles. These are the numbers to lead with.

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  ROC-AUC     💭 │  KS statistic💭 │ Balanced acc 💭 │  Gini        💭 │
│     0.781       │      43.3       │     0.714       │     0.562       │
│ ranks risk      │ separation at   │ fair to both    │ industry        │
│ correctly 78%   │ the best cutoff │ classes         │ convention      │
│ ▲ stable        │ ▲ stable        │ ▲ stable        │ ▲ stable        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

| Metric | Value | Caption (always visible) | 💭 Explanation (on click) |
|---|---|---|---|
| ROC-AUC | 0.781 | ranks risk correctly 78% of the time | Take one customer who defaulted and one who paid. The model gives the defaulter a higher risk score 78 times out of 100. A coin flip would get 50. This is the main number to judge the model by, and it does not change if the proportion of defaulters changes. |
| KS statistic | 43.3 | 58% of defaults caught, 15% of payers touched | The widest gap between the score distributions of defaulters and payers, quoted in percentage points the way credit teams write it. At the score where the two groups separate most, the model has captured 58% of the customers who went on to default while picking up only 15% of those who paid. The gap between those two figures is the KS. Above 40 is strong for a behavioural scorecard, and above 60 usually means the target has leaked into the features. |
| Balanced accuracy | 0.714 | treats both classes as equally important | The average of two things: the share of defaulters we catch (62%) and the share of good payers we correctly leave alone (81%). Because it averages them, a model that ignored defaulters entirely would score 50, not 78. |
| Gini | 0.562 | banking convention for the same thing | Exactly the same information as AUC, rescaled: Gini = 2 × AUC − 1. Credit teams quote this out of habit. 0 is random, 1 is perfect. |

### Secondary metrics

Smaller tiles, collapsible section titled "More metrics".

| Metric | Value | Caption | 💭 Explanation |
|---|---|---|---|
| Weighted F1 | 0.776 | overall quality across both classes | Combines how well we identify defaulters and how well we identify good payers, counting each group by its size. Read it with care: 78% of customers never default, so the easy class carries most of the weight and the figure flatters the model. It is here because papers quote it, not because it is the number to judge us on. |
| Macro F1 | 0.692 | both classes counted equally | Like weighted F1, but small and large groups count the same. Lower than weighted F1 because the minority class is harder. |
| F1, default class | 0.540 | performance on defaulters alone | The hardest single number we report, and the one we do not hide. It looks low, but on this dataset a coin flip scores 0.31 and a naive "was he late last month" rule scores 0.50. Only 22% of customers default, which caps how high this can go. |
| PR-AUC | 0.540 | precision vs recall on defaulters | Similar to AUC but focused entirely on the defaulters. The no skill baseline is 0.221, the default rate, not 0.5. |
| Brier score | 0.135 | are the probabilities honest | Checks whether the numbers mean what they say. If the model says 30%, do about 30% actually default? Lower is better, 0 is perfect. Ours halved after calibration. |
| MCC | 0.393 | correlation with reality | Treats the prediction as one variable and the truth as another and measures correlation. 0 is random, 1 is perfect. Hard to fool with imbalanced data. |
| Cohen kappa | 0.387 | agreement beyond luck | How much better than guessing, after subtracting the agreement you would get by chance. |

Accuracy is deliberately absent. Flagging nobody at all scores 77.9% on this test split,
because that is the share of customers who never default, so the figure says more about the
class balance than about the model and invites exactly the wrong comparison.

KS, Gini, Brier and the F1 variants belong on this screen, whose audience is whoever owns the
model. The loan officer's screen shows the risk band and the reasons behind it: a separation
statistic does not inform any decision they make, and an unlabelled "43.3" invites misreading.

### Confusion matrix

Rendered as a 2×2 heatmap at the live threshold, with counts and row percentages.

```
                    predicted: pays    predicted: defaults
actual: pays             2,926                 578          (84% correctly cleared)
actual: defaults           401                 595          (60% caught)
```

💭 *The two mistakes cost different amounts. The 401 in the bottom left are customers who
defaulted and we did not flag, which costs the outstanding balance. The 578 in the top
right are customers we flagged who would have paid, which costs the interest we would
have earned. A lender usually treats the first as several times more expensive.*

### Risk bands

The table a loan officer actually acts on. Bar chart plus table.

| Band | Share of customers | Actual default rate |
|---|---|---|
| 🟢 Low | 49.7% | 8.8% |
| 🟡 Medium | 32.2% | 21.4% |
| 🔴 High | 18.1% | 59.9% |

💭 *Band edges come from the validation set, never from the test set. The default rates
shown are what actually happened on held out customers the model had never seen.*

### Risk decile chart

Horizontal bar chart, actual default rate per predicted risk decile.

```
decile  1 ███                                         5.1%
decile  2 ███                                         6.2%
decile  3 █████                                       9.3%
decile  4 ███████                                    11.7%
decile  5 ██████████                                 18.3%
decile  6 ██████████                                 17.6%
decile  7 █████████████████                          29.6%
decile  8 ████████████████████████████               47.9%
decile  9 ██████████████████████████████████████████ 70.8%
          ┄┄┄┄┄┄┄┄┄┄┄ overall average 22.1%
```

💭 *This is the clearest evidence the model works. Customers are sorted by predicted risk
and split into ten groups. If the model were guessing, every bar would sit at 22%.
Instead the safest group defaults at 5% and the riskiest at 71%, a 14 times spread.*

---

## Tab 2 · Fairness

### Verdict strip

```
┌────────────────────────────────────────────────────────────────┐
│  ✅ ALL FOUR PROTECTED ATTRIBUTES PASS THE FOUR FIFTHS RULE    │
│  Protected attributes are excluded from model inputs           │
│  Cost of excluding them: 0.0007 AUC                            │
└────────────────────────────────────────────────────────────────┘
```

💭 *The model never sees sex, age, education or marital status when making a decision.
They are kept only to check the results afterwards. Removing them cost almost nothing
in accuracy, so there is no performance argument for putting them back.*

### Disparate impact

One row per attribute, with a gauge showing the 0.80 threshold line.

| Attribute | DIR | Status | Trust |
|---|---|---|---|
| SEX | 0.937 | ✅ Pass | ▲ stable |
| AGE | 0.945 | ✅ Pass | ▲ stable |
| EDUCATION | 0.888 | ✅ Pass | ▲ stable |
| MARRIAGE | 0.988 | ✅ Pass | ▲ stable |

💭 **Disparate impact ratio.** *Do we approve every group at a similar rate? Take the
group approved least often and divide by the group approved most often. 1.00 means
identical treatment. Regulators commonly treat anything below 0.80 as evidence of
adverse impact, which is where the "four fifths" name comes from. EDUCATION at 0.888 is
our closest to the line and worth watching.*

### Equalized odds

Same table, three more columns, each with its own icon. **These carry a warning badge.**

| Attribute | TPR gap | FPR gap | AUC spread | Trust |
|---|---|---|---|---|
| SEX | 0.024 | 0.039 | 0.044 | ⚠ unstable |
| AGE | 0.080 | 0.022 | 0.028 | ⚠ unstable |
| EDUCATION | 0.070 | 0.079 | 0.048 | ⚠ unstable |
| MARRIAGE | 0.019 | 0.025 | 0.000 | ⚠ unstable |

💭 **TPR gap.** *Do we catch real defaulters equally well in every group? 0.080 for age
means one age band has its defaulters spotted 8 in 100 more often than another.*

💭 **FPR gap.** *Do we wrongly flag good customers equally rarely in every group? 0.079
for education means one education group gets wrongly flagged about 8 in 100 more often.*

💭 **AUC spread.** *Is the model equally good at ranking risk inside each group? Marriage
is near zero, meaning the model is equally capable for married and single customers.*

⚠ **Warning banner, displayed above this table:**

> These three columns come from a single data split. Repeating the analysis across five
> independent splits showed the gaps move substantially and sometimes reverse sign. Treat
> them as indicative only. Do not conclude the model is unfair to a specific group from
> these numbers alone. Disparate impact above is stable; these are not.

💭 *Why it matters: it would be easy to read the education row and announce the model
discriminates by education. We tested that and it did not hold up. Honest dashboards say
which of their own numbers are shaky.*

### Per group detail

Expandable panel per attribute. Group, size, actual default rate, approval rate, TPR,
FPR, AUC. Groups with fewer than 50 members or 10 defaults are hidden with a note, since
their rates are too noisy to read.

---

## Tab 3 · Explainability

### Global feature importance

Horizontal bar chart with error bars, top 20 features.

| Feature | AUC drop when shuffled | Plain meaning |
|---|---|---|
| `pay_max` | 0.0275 | worst delinquency in the six months |
| `PAY_1` | 0.0234 | repayment status last month |
| `remaining_credit` | 0.0117 | limit minus current balance |
| *43 others combined* | ~0.026 | everything else |

💭 *Each feature is scrambled in turn and we measure how much the model gets worse. The
bigger the drop, the more the model relied on it. Three features carry almost all the
signal: has the customer missed a payment, how badly, and how much room is left on the
card.*

⚠ *Caveat shown inline: the six monthly payment columns are strongly correlated with each
other, so this method understates each one individually. The honest reading is that the
delinquency group of features dominates, not that exactly three features matter.*

### Single prediction explainer

Input panel on the left, result on the right. Lets the engineer poke the model.

```
┌─ Test a customer ──────┐  ┌─ Result ───────────────────────────────┐
│ Credit limit  200,000  │  │         75.2%   🔴 HIGH RISK           │
│ Last month    2 late   │  │  this band defaults 59.9% of the time  │
│ Balance       190,000  │  ├────────────────────────────────────────┤
│ Payment       0        │  │  Why:                                  │
│ ...                    │  │  ▲ two months behind      +0.31        │
│                        │  │  ▲ no payment made        +0.18        │
│   [ Score customer ]   │  │  ▲ balance 95% of limit   +0.12        │
└────────────────────────┘  │  ▼ long credit history    −0.04        │
                            └────────────────────────────────────────┘
```

💭 *Each row is a SHAP value: how much that single fact pushed this customer's risk up or
down, starting from the average customer. The rows add up to the final score, so the
explanation is complete rather than a summary.*

---

## Tab 4 · Threshold and bands

Interactive. The most useful tab for an ML engineer.

```
Decision threshold  ├────────●──────────────────┤   0.2824
                   0.05                        0.85
```

Dragging it live updates precision, recall, F1, flagged share, the confusion matrix and
expected cost. A marker shows the saved production threshold so you can see how far you
have moved from it.

| Threshold | Flagged | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.20 | 33% | 0.437 | 0.651 | 0.523 |
| **0.27 (live)** | **29%** | **0.477** | **0.622** | **0.540** |
| 0.35 | 20% | 0.566 | 0.523 | 0.544 |
| 0.50 | 11% | 0.677 | 0.340 | 0.453 |
| 0.60 | 9% | 0.703 | 0.276 | 0.397 |

💭 *The threshold is the cut off where a score becomes a decision. Raising it flags fewer
customers, so the ones flagged are more likely to really default (higher precision) but
more defaulters slip through (lower recall). You cannot improve both at once by moving
this slider, only trade them.*

### Cost curve

Second slider for the cost ratio of a missed default versus a wrongly refused customer,
sweeping 1:1 to 10:1, with the cost minimising threshold marked.

💭 *The two mistakes are not equally expensive. Missing a default costs the balance owed.
Refusing a good customer costs the interest you would have earned. Set the ratio your
business believes and the chart shows where to put the threshold.*

### Calibration curve

Predicted probability on the x axis, observed default rate on the y axis, with the
diagonal. Brier score alongside.

💭 *If the model says 30% and 30% of those customers actually default, the dot sits on the
diagonal. Points below the line mean the model is too pessimistic, above means too
optimistic. This matters because the risk bands and any expected loss calculation depend
on the probabilities being truthful, not just correctly ordered.*

---

## Tab 5 · Registry

Table of model versions, newest first.

| Version | Trained | AUC | F1 | Fairness | Status |
|---|---|---|---|---|---|
| lightgbm_ensemble_v1 | 2026-10-03 | 0.781 | 0.540 | ✅ 4/4 | 🟢 Live |
| *(previous versions as they accumulate)* | | | | | |

Actions: view full metrics, promote to live, archive. **A version cannot be promoted
unless its fairness audit passed**, enforced in the UI and again in the API.

💭 *Every model that has ever been trained is recorded with the numbers it achieved and
the data it saw, so a decision made six months ago can be traced back to the exact model
that made it.*

---

## Tab 6 · Drift (build last, cut if time runs short)

Population stability index per feature over time, as a heatmap of features by week.
Thresholds at 0.10 (watch) and 0.25 (significant).

💭 *PSI compares the customers arriving now against the customers the model was trained
on. A high value means the incoming population has shifted and the model may no longer
fit it, and this can be detected before any defaults are even observed.*

This tab needs repeated scoring over time, which the current single snapshot dataset
cannot provide. Treat it as a design placeholder.

---

## Implementation notes

**Use Streamlit.** Python only, no npm, no build step, no second language. The 💭 pattern
is native:

```python
st.metric("ROC-AUC", "0.781", help="Take one customer who defaulted and one who paid...")
```

`st.metric(help=...)` renders a hover icon out of the box. For longer text use
`st.popover("💭")`. For the warning banners use `st.warning`. Tabs are `st.tabs`.

**Everything reads from artifacts, nothing is recomputed in the UI:**

| Panel | Source |
|---|---|
| All performance metrics | `backend/artifacts/scoring_config.json` |
| Risk bands | `scoring_config.json` → `band_report` |
| Fairness tables | `model_metadata.json` → `fairness_summary` |
| Single prediction | `CreditScorer.score_one()` |
| Feature importance | precomputed, saved to `artifacts/feature_importance.json` |
| Threshold sweep | precomputed sweep table, saved as parquet |

The dashboard must never retrain or re-evaluate on request. It loads numbers computed by
`app.ml.train` and cached. That keeps it fast and guarantees the screen agrees with what
was actually evaluated.

**Build order:** Performance → Fairness → Threshold → Explainability → Registry → Drift.
The first three answer most questions and cover the project requirements. The last two
are the ones to drop under time pressure.

## Still needed before this can be built

- `shap` is not installed, so Tab 3's single prediction explainer has no data source yet
- Feature importance is computed in the notebook but not saved as an artifact
- The threshold sweep table is not saved as an artifact
- No model version history exists yet, the registry will show a single row
