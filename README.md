# Behavioural Credit Scorecard System

An explainable, fair credit-risk platform for credit-card portfolios. It scores each
customer's risk of missing their next payment, explains the score in plain language, audits
itself for fairness across demographic groups, and serves everything through a REST API and a
role-based web dashboard.

Final Year Project (TM471), Arab Open University Lebanon.

## Features

- **Risk scoring**: LightGBM ensemble returning a default probability and a Low / Medium /
  High risk band for each customer.
- **Explanations**: per-customer reasons from native tree SHAP, plus LIME and occlusion
  implementations for method comparison.
- **Recourse**: counterfactual "what would help most" suggestions for flagged customers.
- **Fairness audit**: disparate impact and per-group error metrics across sex, age, education
  and marital status, computed on every training run.
- **Role-based dashboard**: separate screens for loan officers, ML engineers and cardholders.
- **Loana AI assistant**: a cardholder and officer chat grounded in the customer's own data,
  backed by OpenAI, a local Ollama model, or fixed templates as a fallback.
- **Monitoring**: population stability index (PSI) drift check and a model registry.
- **Batch scoring**: CSV upload for scoring many customers at once.

## Model at a glance

| | |
|---|---|
| Task | Probability a credit-card customer defaults on their next payment (behavioural scorecard) |
| Data | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/): 30,000 customers, 23 raw features expanded to 46 |
| Model | Three LightGBM fits (seeds 42, 202, 777), averaged |
| Test ROC-AUC | 0.783 (Gini 0.566, KS 43.3) |
| Test F1 / precision / recall | 0.541 / 0.478 / 0.623 at threshold 0.2389 |
| Fairness | Disparate impact 0.88 to 0.99 across the four protected attributes (all pass the four-fifths rule) |

Metrics come from a held-out 4,500-customer test split that informed no modelling decision.
Protected attributes are excluded from the model's inputs and kept only for auditing. Full
results, studies and known limitations are in [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md)
and [docs/REPORT_FINDINGS.md](docs/REPORT_FINDINGS.md).

## Architecture

```text
 Streamlit dashboard  ──HTTP──►  FastAPI service  ──►  PostgreSQL
 (frontend/)                     (backend/app/api)     customers, statements,
                                       │               predictions, decisions, chat
                                       ▼
                                 ML core (backend/app/ml)
                                 scorer · explainability · counterfactuals · drift
                                       │
                                       ▼
                                 backend/artifacts/   (written by train.py)
```

- The **API** is routing only; scoring maths lives in `app/ml`.
- The **dashboard** never recomputes metrics. `train.py` writes every artifact and the API and
  UI read them, so what is shown is exactly what was evaluated.
- **Sessions**: `POST /session` verifies a password and returns a signed, expiring bearer
  token. The role is read from the account, never from the request.

### Roles

| Role | Can do |
|---|---|
| **Loan officer** | Work the risk queue, open a customer, see band, reasons and history, approve / decline / refer with a justification, upload a CSV batch, chat with Loana |
| **ML engineer** | View performance, fairness, explanations, an interactive threshold, drift and the model registry |
| **Cardholder** | See their own status, a simplified reason and what would help, and chat with Loana. Data is reduced server-side, so what a role cannot see is never sent |

## Tech stack

| Layer | Tools |
|---|---|
| Modelling | LightGBM, scikit-learn, XGBoost, PyTorch (comparison studies), LIME, Fairlearn |
| API | FastAPI, Uvicorn, SQLAlchemy 2, Pydantic |
| Database | PostgreSQL 16 (psycopg 3) |
| Dashboard | Streamlit, Plotly |
| Assistant | OpenAI `gpt-4o-mini`, or Ollama `qwen2.5:3b` |
| Tooling | Docker Compose, pytest, ruff, black, isort |

## Getting started

### Prerequisites

- Docker with Compose
- Python 3.11 (only needed to train the model locally)
- The dataset: download *default of credit card clients.xls* from the
  [UCI repository](https://archive.ics.uci.edu/dataset/350/) into `data/raw/`
  (`data/` is gitignored)

### 1. Configure

Create a `.env` file in the repository root:

```bash
DB_PASSWORD=choose-a-password
SESSION_SECRET=<random string>      # python -c "import secrets; print(secrets.token_urlsafe(32))"
DEMO_MODE=1                         # creates the demo accounts below
```

| Variable | Required | Purpose |
|---|---|---|
| `DB_PASSWORD` | yes | Password for the Compose Postgres |
| `SESSION_SECRET` | yes | Key that signs session tokens |
| `DEMO_MODE` | no (default `1`) | `1` seeds one demo account per role |
| `SEED_PASSWORD` | no | Password for seeded staff accounts |
| `WEB_PORT` | no (default `8502`) | Dashboard port |
| `OPENAI_API_KEY` | no | Enables the OpenAI-backed assistant |
| `OPENAI_MODEL` | no (default `gpt-4o-mini`) | OpenAI model name |
| `OLLAMA_URL`, `OLLAMA_MODEL` | no | Local fallback model for the assistant |

### 2. Train the model

Generates every artifact the API and dashboard read:

```bash
python -m venv .venv
.venv/Scripts/activate              # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cd backend
python -m app.ml.train
```

### 3. Run the stack

```bash
docker compose up --build
```

Open <http://localhost:8502>. Compose starts Postgres, loads the dataset and demo accounts
(`seed`), then the API (port 8014 inside the network) and the dashboard. The code is mounted
into the containers, so saved changes reload without a rebuild; rebuild only after changing
`requirements.txt`.

### Demo accounts

With `DEMO_MODE=1` every role has an account on the password `demo`:

| Role | Email |
|---|---|
| Loan officer | `officer@demo.test` |
| ML engineer | `engineer@demo.test` |
| Cardholder | `cardholder@demo.test` |

Never enable `DEMO_MODE` outside a demo.

### Loana AI and your data

With `OPENAI_API_KEY` set, the assistant uses OpenAI, so data leaves the machine: the fact
sheet it sends holds the risk band, the reasons, account figures and six months of history,
never the probability or any protected attribute. Without a key it falls back to a local
[Ollama](https://ollama.com) model (`ollama pull qwen2.5:3b`), and without that to templates,
so the chat always answers. No Ollama on the host? Start the stack with
`docker compose --profile ollama up` and set `OLLAMA_URL=http://ollama:11434`.

## API

Interactive docs are served by FastAPI at `/docs` on the API host. Main route groups:

| Group | Routes |
|---|---|
| Session | `POST /session`, `GET /session` |
| Officer | `GET /officer/queue`, `GET /officer/customers/{id}`, decisions, `POST /officer/customers/{id}/chat` |
| Cardholder | `GET /cardholders/...`, assistant chat |
| Predictions | `POST /predictions`, `GET /predictions/{id}`, `POST /predictions/batch` |
| Decisions | `POST` and `GET` under `/predictions/{id}/decisions`, `GET /decisions` |
| Model | `GET /report`, `/report/threshold-sweep`, `/report/cost-curve`, `/report/feature-importance`, `/segments`, `/model-versions`, `POST /explain` |
| Fairness | `GET /fairness` |
| Monitoring | `GET /monitoring/drift`, `/monitoring/performance`, `/monitoring/assistant` |
| Recourse | counterfactual suggestions per customer |
| Service | `GET /health` |

## Testing

```bash
cd backend
python -m pytest
```

Covers the API, role boundaries, sessions, the database schema, explanations, counterfactuals
and the assistant. Dashboard tests live in `frontend/tests/`.

## Project structure

```text
credit_risk_system/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers (routing only)
│   │   ├── ai/           # Loana assistant, LLM clients, coaching text
│   │   ├── core/         # Config, password hashing, session tokens
│   │   ├── db/           # SQLAlchemy models, ingestion, seed
│   │   ├── ml/           # Training, scoring, explanations, fairness, drift, studies
│   │   ├── schemas/      # Pydantic request and response shapes
│   │   └── main.py       # FastAPI app and /health
│   ├── artifacts/        # Generated by train.py (gitignored)
│   └── tests/
├── frontend/             # Streamlit app: views/ (officer, cardholder, sign-in), tabs/ (engineer)
├── notebooks/            # 01 data prep, 02 modelling, 03 segmentation, 04 studies, 05 multiplicity
├── docs/                 # Status, conventions, research notes, design spec
├── data/raw/             # Dataset (not committed)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

### Regenerating study artifacts

Each study module writes one file to `backend/artifacts/`:

```bash
python -m app.ml.experiments            # model comparison
python -m app.ml.segmentation           # behavioural segments
python -m app.ml.clustering_comparison
python -m app.ml.multiplicity           # predictive multiplicity
```

Run the notebooks in numeric order on the **Python (credit-risk)** kernel
(`python -m ipykernel install --user --name credit-risk --display-name "Python (credit-risk)"`).

## Documentation

- [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md): scope, research question, studies, method, known gaps
- [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md): definition of done and current status
- [docs/CODE_CONVENTIONS.md](docs/CODE_CONVENTIONS.md): code style and the artifact contract
- [docs/frontend_ml_engineer_view.md](docs/frontend_ml_engineer_view.md): ML engineer screen design

## References

- Yeh, I. C. and Lien, C. H. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473–2480.
- Lessmann, S., Baesens, B., Seow, H. V. and Thomas, L. C. (2015). Benchmarking
  state-of-the-art classification algorithms for credit scoring. *European Journal of
  Operational Research*, 247(1), 124–136.
