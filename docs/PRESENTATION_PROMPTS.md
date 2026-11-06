# Presentation prompt set

Prompts for the final build-out before the FYP II presentation. Revised against what the
repo actually contains on 2026-11-04.

**What changed from the first draft.** The original five prompts were written as if the
system did not exist yet. It does: the database, the API, the ML core, the role routing and
the Streamlit screens are all built. The prompts below are re-scoped to the work that is
genuinely missing, and each one now carries the repo's real names, the real scale, and an
acceptance check.

---

## Rules that apply to every prompt

Paste this block once at the top of a session, or rely on the per-prompt pointer.

```
Before writing any code, read docs/CODE_CONVENTIONS.md and docs/PROJECT_STATUS.md.
Match the existing code; do not introduce a second way of doing something that is
already done one way.

Non-negotiables for this repo:
- `backend/app/ml/feature_engineering.py` is the single source of truth. Training and
  serving both import `engineer()` from it. Never reimplement feature logic elsewhere.
- `train.py` writes `backend/artifacts/`; everything else only reads. The UI and API must
  never recompute a metric that is already in `scoring_config.json`.
- Logic lives in `ml/`, routing lives in `api/`. No scoring maths in a route handler.
- `shap` is deliberately not a dependency. Local explanations come from LightGBM's native
  `predict(X, pred_contrib=True)` via `app.ml.explainability.Explainer`, which is exact
  TreeSHAP for tree models. Do not add the `shap` package or `TreeExplainer`.
- Protected attributes (sex, age, education, marital status) are accepted as input for
  auditing and dropped inside `CreditScorer.prepare()` before scoring.
- Never hardcode credentials. `DATABASE_URL` and any API key come from `.env` via
  `app.core.config`.

Before you write code, reply with: what already satisfies this request, what is genuinely
missing, and the smallest change that closes the gap. Wait for my go-ahead.
```

That last paragraph is the highest-value line in this document. It is what stops a model
from rebuilding `db/models.py` because the prompt said "implement data models".

### The scale, stated correctly

The model outputs a **calibrated probability of default next month**, not a 300–850 score.
`scoring_config.json` records `"scale": "probability of default next month"`.

| Quantity | Value |
|---|---|
| Decision threshold | 0.2389 |
| Band edges | Low < 0.1471, Medium < 0.3312, else High |
| Observed default rate by band | Low 8.9%, Medium 20.3%, High 56.3% |
| Test ROC-AUC / Gini / KS | 0.783 / 0.566 / 43.3 |
| Features | 46, listed in `scoring_config.json` |
| Model version | `lightgbm_ensemble_v1` |

300–850 is a FICO-style **application** scorecard convention. This is a **behavioural**
scorecard on the UCI Taiwan data, and there is no derivation behind a 300–850 mapping here.
If an examiner asks how the scale was calibrated, "probability of default, with band edges
chosen on validation" is an answer; "300–850" is not. If you want a score-like dial for the
customer screen, make it an explicitly labelled monotone transform of the probability — and
say on screen that it is a presentation device.

---

## Prompt 0 — Re-ground the status document

Run this first. Everything else inherits its accuracy, and you will present from it.

```
Act as the engineer who has to defend this repo in a viva.

docs/PROJECT_STATUS.md is dated 2026-10-04 and is now stale. It lists the FastAPI service,
the loan officer screen, the ML engineer dashboard, the applicant role, LIME,
counterfactuals, bias mitigation and the explanation stability comparison as not started.
Several of those are built.

Audit the repo against that status table:
- backend/app/api/, backend/app/ml/, backend/app/db/, backend/tests/
- frontend/app.py, frontend/views/, frontend/tabs/
- backend/artifacts/ and what train.py actually writes
- notebooks/01 to 05

Then rewrite section 3 (Status) and section 4 (What remains) to match reality, and update
the verified-results table in section 2 from backend/artifacts/scoring_config.json rather
than from memory — the stored test metrics are ROC-AUC 0.7830, Gini 0.5661, KS 43.33,
F1 0.5405 at threshold 0.2389, and the document currently quotes slightly different
figures from an earlier run.

Keep section 6 (Honesty register) and add to it anything you find that a reader would be
misled about. Do not delete entries.

Report any place where the document claims something the code does not do, before you edit.
```

Why this matters: the status doc and the artifact disagree on the headline numbers
(0.781 vs 0.783 AUC, threshold 0.2747 vs 0.2389). Presenting one while the code says the
other is the single easiest thing for an examiner to catch.

---

## Prompt 1 — Counterfactuals (replaces the original "Layer 1: database")

The database and ingestion API exist. The real objective-2 gap is counterfactuals: no
module computes them. This is also the engine the coaching layer needs, so it comes first.

```
Act as a Lead ML Engineer. Add a counterfactual explanation module at
backend/app/ml/counterfactuals.py, following the interface style already set by
backend/app/ml/explanation_methods.py (ExplanationMethod / Explanation, dataclass with
to_dict()).

Question it answers: what is the smallest change to this customer's own behaviour that
moves them below the decision threshold (0.2389), or down one risk band.

Requirements:
- Search only over features the customer could plausibly act on next month: payment
  amounts (PAY_AMT1), and the engineered quantities they drive — utilisation, repayment
  ratio, remaining credit. Treat LIMIT_BAL and payment history older than last month as
  fixed, and never propose changing a protected attribute.
- Respect the derivation chain: changing PAY_AMT1 must flow through engineer() so the
  engineered features stay consistent. Perturbing an engineered column directly produces a
  customer who cannot exist.
- Return the required change in raw units, the resulting probability and band, and whether
  a feasible counterfactual was found at all. "No single-month action reaches Low" is a
  valid and important result — report it rather than returning the nearest miss as if it
  succeeded.
- Deterministic. Seeded if any search randomness is involved, per conventions section 11.
- Expose it through the API the way the other explanation work is exposed, and add it to
  the engineer's explanation tab for comparison against SHAP, LIME and occlusion.

Acceptance check, in the style of conventions section 14: run it on
app.ml.scorer.example_customer(), assert the counterfactual's recomputed probability
actually falls below the threshold when one is reported, and assert the recomputed
probability comes from CreditScorer rather than from the search's own estimate.

Add a short module docstring recording the feasibility constraint and why engineered
features are not perturbed directly.
```

---

## Prompt 2 — Coaching built on facts, not on the model's imagination

Your original Layer 3 asked an LLM to turn SHAP values into advice. Keep the idea; change
where the numbers come from. An LLM that computes or guesses a recovery target will
eventually state a number the model does not support, in front of your supervisor.

```
Act as an AI Product Engineer. Add backend/app/ml/coaching.py — a text layer over
explanations that are already computed.

Hard contract: the LLM re-words facts, it never produces them. Every number and every
action in the output must come from a structured payload assembled before the call:
- the calibrated probability and risk band from CreditScorer
- the top risk-raising reasons from Explainer.reasons(), already in plain language via
  describe()
- the actionable target from counterfactuals.py, if one was found

Requirements:
- A function that takes that payload and returns coaching text. The template path is the
  primary implementation and is built first; it must be good enough to demo on its own.
  An LLM call is an optional refinement that may only rephrase the template's facts. If no
  API key is configured, or the call fails or times out (cap it at a few seconds), return
  the template result. The demo must never wait on a network.
- Read the API key from .env via app.core.config. Never inline it. Add the dependency to
  requirements.txt pinned, per conventions section 15.
- No guarantees. "Paying 15,000 next month would move your risk to Medium on this model"
  is honest; "this will fix your credit" is not. No financial advice framing, no promise
  about any real lender's decision, and a visible line saying this is a model's estimate
  on a prototype.
- If no feasible counterfactual exists, say what the model is responding to and that no
  single-month action clears it. Do not invent an achievable-sounding target.
- User-facing wording follows conventions section 12: it belongs in this module, not
  scattered through the Streamlit page.

Then pre-generate coaching for the demo cardholders in db/seed.py and cache it to
backend/artifacts/, so the presentation renders from a file and is identical every run.
Keep the live path available to show it working.

Acceptance check: with the key unset, assert the function still returns non-empty text and
that every number in it appears in the input payload.
```

The pre-generation step is the difference between a feature and a demo that survives
university wifi.

---

## Prompt 3 — Role routing: make the honesty visible (replaces "Layer 4: authentication")

This already exists, and the design decision behind it is deliberate and documented. Do
not let a prompt talk you into a fake login.

`backend/app/api/roles.py` and `frontend/app.py` both already state that role selection is
not authentication, and conventions section 1 says there is no `auth.py` on purpose: *"a
real deployment puts an identity provider in front of the API, and saying so is more honest
than a login form that checks nothing."*

Keep the role names as they are: **officer**, **engineer**, **cardholder**. The original
draft renamed them to Customer / Underwriter / Risk Manager — that rename would touch the
API, the DB role vocabulary (`ROLES` in `db/models.py`), five test files, both docs and
your report, and buy nothing.

```
Act as a Principal Software Architect. Review, do not rebuild, the role layer in
backend/app/api/roles.py, backend/app/api/dependencies.py, backend/app/db/models.py and
frontend/app.py.

The system deliberately has roles without authentication, and says so. Keep that. What I
want is for it to be airtight and legible to an examiner:

- Confirm the API, not the page, decides what each role is sent. Report any screen that
  draws data the API would not have given that role, and any endpoint that returns a
  cardholder's data without the role being checked.
- Confirm one cardholder cannot read another's record by changing an id, and say plainly
  what currently prevents it and what does not.
- Make the "this is not authentication" statement visible on the role-selection screen and
  in the OpenAPI description, in one sentence, with the production answer (identity
  provider in front of the API) named.
- Add a test to backend/tests/test_api_roles.py for whatever gap you find.

Report findings before changing anything. If the answer is that the layer is already
correct, say so and only add the test and the visible statement.
```

---

## Prompt 4 — Surface the research angle (missing from the original plan entirely)

Your five layers build a product. They do not show the research contribution, which is
what the FYP is marked on: *explanation methods compared for consistency*.
`explanation_methods.py` already implements three methods and `run_to_run_stability()`.
`mitigation.py` already computes the fairness-cost frontier. Neither should be hidden
behind a slide.

```
Act as a Research Engineer preparing a defence. The project's research angle is whether
explanation methods are consistent. backend/app/ml/explanation_methods.py implements
TreeShap, Lime and Occlusion behind one interface plus jaccard() and
run_to_run_stability(); backend/app/ml/mitigation.py computes what each bias mitigation
costs in discrimination.

Make both presentable in the ML engineer view (frontend/tabs/explain.py and
frontend/tabs/fairness.py):

- One screen answering: do the methods agree on the same customer, and does each method
  agree with itself across runs. Show top-k overlap between methods, LIME's run-to-run
  variance against SHAP's exactness, and stability under the three ensemble seeds. The
  expected result — SHAP perfectly stable because it is exact, LIME not because it samples
  — is the finding; state it on the screen in one sentence.
- One screen for the mitigation frontier: each method's position on discrimination against
  fairness, with the cost of each intervention named.
- Read from artifacts written by the modules. Do not recompute on page load, per
  conventions section 3. If an artifact the screen needs does not exist yet, add the module
  run that writes it and document the command in PROJECT_STATUS.md section 7.
- Every figure gets a plain-language help string beside it, the way the performance tab
  already does with st.metric(help=...).

Do not present the equalised-odds TPR/FPR/AUC gaps as findings. The honesty register
records that they move between splits and sometimes reverse sign. Disparate impact is the
stable measure; all four attributes pass four-fifths — marriage 0.990, sex 0.938,
age 0.938, education 0.880.
```

That last paragraph protects you from your own strongest-looking result. The sex AUC gap
was tested and killed — it did not replicate across five splits. Keep it as methodology,
never as a finding.

---

## Prompt 5 — Presentation polish and demo hardening (replaces "Layer 5: build the frontend")

The frontend exists: `app.py`, two views, six tabs, a theme module, 1,400 lines. It needs
hardening and a last visual pass, not a rewrite.

```
Act as an Expert UI/UX Engineer doing a final pass on an existing Streamlit app before a
live university presentation. Do not restructure what works. frontend/theme.py already
sets the visual language and .streamlit/config.toml the palette — extend them rather than
introducing a parallel styling approach.

Three screens, three jobs:
- Cardholder (frontend/views/cardholder.py): where they stand, the reasons in plain
  language, the coaching card from app.ml.coaching, and the counterfactual target if one
  exists. No jargon, no feature names, no raw probabilities presented as certainty.
- Loan officer (frontend/views/officer.py): the worklist, one customer's reasons, the
  decision with a justification box. Optimise for how fast a reason can be read aloud.
- ML engineer (frontend/tabs/): performance, fairness, explanation stability, threshold,
  drift, registry. Dense is fine here; this is the audit view.

Hardening, which matters more than polish on the day:
- Every screen must render something meaningful if the API is down or an artifact is
  missing — a stated reason, not a traceback.
- Fix the cold-start path: first load must not block on anything that can be cached.
- Check every number on screen against backend/artifacts/. Report any screen showing a
  figure that disagrees with the artifact, before changing styling.
- Check it at the resolution you will actually project at, and that charts are legible from
  the back of a room — font sizes, not just colours.

Report what you found disagreeing with the artifacts first. That list is more valuable to
me than the visual pass.
```

---

## Prompt 6 — Adversarial review before the viva

Run this last, in a fresh session so it has no memory of having built anything.

```
Act as an external examiner who has not seen this project and is sceptical of it. Read
docs/PROJECT_STATUS.md, docs/REPORT_FINDINGS.md, docs/CODE_CONVENTIONS.md, the notebooks
and backend/app/ml/.

Produce the ten hardest questions you would ask in a viva, hardest first, each with the
strongest honest answer the repo supports — and mark any question where the repo does not
support a good answer yet.

Probe at least these:
- Why 0.78 and not higher, when published papers on this dataset report more
- Whether the target is circular, given the model sees recent payment status
- Why accuracy is never reported
- Why protected attributes are excluded if excluding them costs almost nothing
- Why tuning 70 configurations changed nothing, and what that says about the approach
- Whether role selection without authentication is a limitation or a cop-out
- Whether an LLM coaching layer adds anything a template could not
- What the explanation-consistency finding actually establishes

Do not be generous. Where an answer is weak, say so and name the smallest piece of work
that would strengthen it.
```

---

## Order to run them

| # | Prompt | Why here |
|---|---|---|
| 0 | Re-ground status | Everything downstream quotes it, and it is currently wrong |
| 1 | Counterfactuals | Closes an objective-2 gap and is prompt 2's input |
| 2 | Coaching | Needs 1; pre-generate for the demo |
| 3 | Role layer review | Cheap, mostly confirmation, add tests |
| 4 | Research angle | The marked contribution; highest value per hour |
| 5 | Polish and hardening | Last, so it hardens the finished thing |
| 6 | Adversarial review | Fresh session, after everything exists |

If time runs out, prompts 0, 1 and 4 are the ones that change your mark. Prompt 2 is the
one that impresses in the room. Prompt 5 is the one that makes the room trust the rest.

---

## What was dropped from the original set, and why

| Original | Verdict |
|---|---|
| Layer 1: build DB schema and ingestion API | Exists: `db/models.py`, `database.py`, `seed.py`, nine API modules, `test_db_schema.py` |
| Layer 2: load model, output 300–850 score | Scoring exists (`CreditScorer`). The 300–850 scale is undefined for a behavioural scorecard and was removed |
| Layer 2: integrate TreeExplainer / SHAP | Contradicts conventions section 15. Native `pred_contrib` is already exact for tree models; `shap` is deliberately not a dependency |
| Layer 2: implement ROC-AUC, KS, disparate impact | Already computed in `train.py` and stored in artifacts. Recomputing in the UI violates the artifact contract |
| Layer 3: LLM coaching from SHAP values | Kept, re-scoped: facts from the counterfactual engine, LLM only re-words, template path primary |
| Layer 4: authentication and secure sessions | The absence of auth is a documented, defensible decision. Re-scoped to a review plus visible honesty |
| Layer 4: Customer / Underwriter / Risk Manager | Renaming officer/engineer/cardholder would break the API, the DB role vocabulary, five tests, both docs and the report |
| Layer 5: build the frontend | Exists, 1,400 lines across two views and six tabs. Re-scoped to hardening and a visual pass |
| "SOLID, 1–2 line docstrings, no credentials" in every prompt | Replaced by a pointer to `docs/CODE_CONVENTIONS.md`, which is concrete. Repeating "SOLID" at a model invites abstraction the project does not need |
| — | Added: status re-grounding, counterfactuals, the research angle, acceptance checks, and an adversarial review |
