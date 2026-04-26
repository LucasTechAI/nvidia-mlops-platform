<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-MLOps_Platform-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA MLOps Platform" />
</p>

<h1 align="center">🟢 NVIDIA MLOps Platform</h1>

<p align="center">
  <strong>End-to-end ML platform for stock price prediction with LSTM, real-time monitoring, LLM-powered agent, and a modern Next.js dashboard.</strong>
</p>

<p align="center">
  <a href="https://github.com/LucasTechAI/nvidia-mlops-platform/actions/workflows/ci.yml"><img src="https://github.com/LucasTechAI/nvidia-mlops-platform/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-14-000000?logo=next.js&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/MLflow-3.5%2B-0194E2?logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/Optuna-4.8%2B-3366CC" alt="Optuna" />
  <img src="https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white" alt="Prometheus" />
  <img src="https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white" alt="Grafana" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License" />
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#%EF%B8%8F-dashboard">Dashboard</a> •
  <a href="#-api-66-endpoints">API</a> •
  <a href="#-ai-agent">AI Agent</a> •
  <a href="#-testing">Testing</a> •
  <a href="#-documentation">Docs</a>
</p>

---

## 🎯 Overview

A production-grade MLOps platform that predicts **NVIDIA (NVDA) stock closing price** for the next 30 days using deep learning. Built with a full ML lifecycle: ETL, training, HPO, serving, monitoring, evaluation, and an AI agent — all orchestrated through **6 microservices** and a **modern dashboard**.

> **FIAP Post-Tech MLET** — Tech Challenge Fase 5

### ✨ Key Features

| Category | Features |
|----------|----------|
| **🧠 Model** | Stacked LSTM (PyTorch) · 30-day forecast · Confidence intervals · Champion-Challenger |
| **⚡ API** | FastAPI with 66 endpoints · Swagger / ReDoc · Async training · Background tasks |
| **🖥️ Dashboard** | Next.js 14 · 11 pages · Dark theme · Real-time charts (Recharts) · Responsive |
| **🤖 AI Agent** | ReAct agent · RAG with ChromaDB · OpenRouter LLM · Markdown responses |
| **📊 Monitoring** | Prometheus · Grafana · Drift detection · SLA tracking · Telemetry |
| **🔬 Evaluation** | RAGAS · LLM-as-Judge · LIME explainability · Golden set (25 entries) |
| **🔧 MLOps** | MLflow tracking · Optuna HPO · Model registry · DVC pipeline |
| **🐳 Deploy** | Docker Compose · Multi-service · Nginx reverse proxy · CI/CD |
| **🔒 Security** | Input guardrails · PII detection · OWASP LLM Top 10 · Bandit + pip-audit |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA MLOps Platform                             │
│                                                                          │
│   ┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐     │
│   │ Yahoo       │────▶│ ETL Pipeline │────▶│ SQLite DB (6700+)    │     │
│   │ Finance     │     │ (yfinance)   │     │ + Feature Store      │     │
│   └─────────────┘     └──────────────┘     └──────────┬───────────┘     │
│                                                        │                 │
│   ┌────────────────────────────────────────────────────┼──────────┐     │
│   │                  Training Pipeline                  │          │     │
│   │   ┌──────────────┐  ┌───────────┐  ┌────────────┐ │          │     │
│   │   │ Preprocessing│─▶│ Sequence  │─▶│ LSTM Model │ │          │     │
│   │   │ MinMaxScaler │  │ Generator │  │ 2×128 units│ │          │     │
│   │   └──────────────┘  └───────────┘  └─────┬──────┘ │          │     │
│   │                                          │         │          │     │
│   │   ┌──────────┐   ┌──────────────┐  ┌────▼──────┐  │          │     │
│   │   │ Optuna   │──▶│ Champion vs  │─▶│ Model     │  │          │     │
│   │   │ HPO (50+)│   │ Challenger   │  │ Registry  │  │          │     │
│   │   └──────────┘   └──────────────┘  └───────────┘  │          │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│   ┌──────────────────── Services (6) ────────────────────────────┐     │
│   │                                                               │     │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │     │
│   │  │ FastAPI    │  │ Next.js    │  │  MLflow    │             │     │
│   │  │ REST API   │  │ Dashboard  │  │  Tracking  │             │     │
│   │  │ :8000      │  │ :3001      │  │  :5000     │             │     │
│   │  └────────────┘  └────────────┘  └────────────┘             │     │
│   │                                                               │     │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │     │
│   │  │ Prometheus │  │  Grafana   │  │  Optuna    │             │     │
│   │  │ Metrics    │  │ Dashboards │  │  Dashboard │             │     │
│   │  │ :9090      │  │ :3000      │  │  :8080     │             │     │
│   │  └────────────┘  └────────────┘  └────────────┘             │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│   ┌──────────────────── AI & Evaluation ─────────────────────────┐     │
│   │                                                               │     │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │     │
│   │  │ ReAct      │  │ ChromaDB   │  │ OpenRouter │             │     │
│   │  │ Agent      │◀▶│ Vector DB  │  │ Gemini 2.0 │             │     │
│   │  └────────────┘  └────────────┘  └────────────┘             │     │
│   │                                                               │     │
│   │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │     │
│   │  │ RAGAS      │  │ LLM-Judge  │  │ LIME       │             │     │
│   │  │ Evaluation │  │ Scoring    │  │ Explainer  │             │     │
│   │  └────────────┘  └────────────┘  └────────────┘             │     │
│   │                                                               │     │
│   └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| Node.js | 18+ |
| PyTorch | ≥ 2.6.0 |
| CUDA (optional) | 12.x |
| Docker + Compose | 24.x+ |
| RAM | 4 GB minimum |

### 1. Clone & Install

```bash
git clone https://github.com/LucasTechAI/nvidia-mlops-platform.git
cd nvidia-mlops-platform

# Python backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Next.js frontend
cd dashboard-frontend && npm install && cd ..
```

### 2. Configure Environment

```bash
cp .env.example .env
```

```env
# .env
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemini-2.0-flash-001
OPENROUTER_API_KEY=your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
```

### 3. Run the Full Pipeline

```bash
# ETL → Training → HPO → Prediction → Start all 6 services
bash scripts/run_all.sh
```

Or step by step:

```bash
python3 scripts/run_etl_nvidia.py       # ETL: Yahoo Finance → SQLite
bash scripts/run_training.sh             # Train LSTM model
bash scripts/run_hpo.sh 20               # Optuna HPO (20 trials)
bash scripts/run_prediction.sh           # Generate 30-day forecast
bash scripts/run_services.sh             # Start all 6 services
```

### 4. Open the Dashboard

| Service | URL | Description |
|---------|-----|-------------|
| 🖥️ **Dashboard** | http://localhost:3001 | Next.js main interface |
| ⚡ **API Docs** | http://localhost:8000/docs | Swagger UI (66 endpoints) |
| 📖 **ReDoc** | http://localhost:8000/redoc | Alternative API docs |
| 📊 **MLflow** | http://localhost:5000 | Experiment tracking |
| 📈 **Prometheus** | http://localhost:9090 | Metrics & targets |
| 📉 **Grafana** | http://localhost:3000 | Monitoring dashboards |
| 🔬 **Optuna** | http://localhost:8080 | HPO visualization |

### Makefile Shortcuts

```bash
make help          # List all commands
make data          # Run ETL
make train         # Train model
make hpo           # Optimize hyperparameters
make serve         # Start API
make dashboard     # Start dashboard
make mlflow-ui     # Start MLflow UI
make test          # Run tests
make lint          # Check code
make all           # Lint + typecheck + security + tests
```

---

## 🖥️ Dashboard

Modern **Next.js 14** dashboard with **11 pages** and dark NVIDIA-themed UI:

| Page | Route | Description |
|------|-------|-------------|
| 🏠 **Home** | `/home` | Overview with key metrics, current price, daily change |
| 📈 **Predictions** | `/predictions` | 30-day forecast chart with confidence intervals |
| 📊 **Metrics** | `/metrics` | RMSE, MAE, MAPE, loss curves, training history |
| 🔬 **Evaluation** | `/evaluation` | 3 tabs: Model Metrics · Explainability (LIME) · LLM Evaluation (RAGAS + LLM-Judge) |
| 👁️ **Observability** | `/observability` | 4 tabs: Drift Detection · Champion-Challenger · Model History · Telemetry (6 services) |
| ⚙️ **MLOps** | `/mlops` | 6 tabs: Model Registry · Business Metrics · SLA · Training Config · P&L Tracking · Cost Analysis |
| 🤖 **AI Agent** | `/agent` | Chat with ReAct agent (RAG + tools), Markdown responses, active model card |
| 🏗️ **Architecture** | `/architecture` | System architecture diagram |
| 📋 **Model Schema** | `/model-schema` | Model card with architecture details |
| 📝 **Logs** | `/logs` | Application and service logs viewer |
| 🚀 **Next Steps** | `/next-steps` | Roadmap with 10 planned features, filterable by category and priority |

### Dashboard Tech Stack

- **Framework**: Next.js 14 (App Router) + TypeScript
- **Styling**: Tailwind CSS + custom dark theme
- **Charts**: Recharts (Line, Bar, Area, Pie, Radar)
- **Icons**: Lucide React (60+ icons)
- **Markdown**: react-markdown + @tailwindcss/typography
- **State**: React hooks (useState, useEffect)

---

## ⚡ API (66 Endpoints)

FastAPI REST API at **http://localhost:8000** with automatic Swagger documentation.

### Core Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Health check (status, uptime, llm_model, llm_provider) |
| `GET` | `/health/ready` | Readiness probe for orchestration |
| `GET` | `/data` | Historical NVIDIA data |
| `GET` | `/data/summary` | Summary statistics |
| `POST` | `/predict` | Generate N-day forecast with confidence intervals |
| `POST` | `/predict/inference` | Inference on custom sequence |
| `POST` | `/train` | Start async training (background) |
| `POST` | `/train/sync` | Synchronous training (blocking) |
| `GET` | `/train/status` | Training progress |

### Agent & Evaluation

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/agent/chat` | Chat with ReAct AI agent |
| `GET` | `/evaluation/ragas` | Run RAGAS evaluation on golden set |
| `GET` | `/evaluation/llm-judge` | LLM-as-Judge scoring |
| `GET` | `/evaluation/lime` | LIME feature importance |

### MLOps & Monitoring

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/mlops/model-registry` | Model versions & status |
| `GET` | `/mlops/business-metrics` | Business KPIs |
| `GET` | `/mlops/sla` | SLA compliance metrics |
| `GET` | `/mlops/pnl-history` | P&L tracking history |
| `GET` | `/mlops/cost-analysis` | Infrastructure & LLM cost breakdown |
| `GET` | `/monitoring/drift` | Data / concept drift detection |
| `GET` | `/monitoring/champion-challenger` | Model comparison results |

### Usage Examples

```bash
# Health check
curl http://localhost:8000/health

# 30-day forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 30, "confidence_level": 0.95}'

# Chat with AI agent
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the predicted trend for NVIDIA?"}'

# Start training
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 32, "learning_rate": 0.001}'
```

---

## 🤖 AI Agent

Interactive **ReAct agent** with Retrieval-Augmented Generation:

| Component | Technology |
|-----------|------------|
| **LLM** | Google Gemini 2.0 Flash via OpenRouter |
| **Vector Store** | ChromaDB with sentence-transformers (`all-MiniLM-L6-v2`) |
| **Framework** | LangChain ReAct agent with custom tools |
| **Tools** | Stock data lookup · Prediction trigger · Metric retrieval |
| **Output** | Markdown-formatted responses with analysis |

```
User: "What's the NVIDIA stock prediction for next week?"
Agent: Retrieves data → Runs prediction → Generates formatted analysis with confidence intervals
```

---

## 🧠 Model

### LSTM Architecture

| Parameter | Value |
|-----------|-------|
| Type | Stacked LSTM (`nn.LSTM`) |
| Layers | 2 |
| Hidden Size | 128 |
| Dropout | 0.2 |
| Sequence Length | 60 days (lookback window) |
| Features | OHLCV (Open, High, Low, Close, Volume) |
| Output | 1 (next-day closing price) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 (early stopping, patience=10) |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE |
| Split | 70% train / 15% val / 15% test |
| Normalization | MinMaxScaler (0–1) |

### Performance

| Metric | Normalized (0–1) | Real-world (USD) |
|--------|-----------------|-----------------|
| **R²** | — | **0.940** |
| **RMSE** | 0.053 (val) · 0.138 (test) | 8.23 |
| **MAE** | 0.031 (val) · 0.080 (test) | 5.87 |
| **MAPE** | — | 2.1% |

> Dataset: ~6,700 records (2017–2026), 70% train / 15% val / 15% test. Normalized metrics computed on MinMaxScaler (0–1) output; real-world metrics computed after inverse transform.

### Hyperparameter Optimization

Bayesian search via **Optuna** with TPE Sampler:

| Hyperparameter | Search Space |
|----------------|-------------|
| `num_layers` | [1, 2, 3, 4] |
| `hidden_size` | [32, 64, 128, 256] |
| `learning_rate` | [1e-5, 1e-2] (log) |
| `dropout` | [0.1, 0.5] |
| `sequence_length` | [30, 60, 90, 120] |
| `batch_size` | [16, 32, 64, 128] |

```bash
bash scripts/run_hpo.sh 50    # 50 trials
```

---

## 🔬 Evaluation

### RAGAS (Retrieval-Augmented Generation Assessment)

Evaluates the AI agent's RAG pipeline quality using OpenRouter + local HuggingFace embeddings:

| Metric | Score | Description |
|--------|-------|-------------|
| Faithfulness | 0.461 | Answer grounded in retrieved context |
| Answer Relevancy | 0.570 | Response relevance to question |
| Context Precision | 0.940 | Retrieved context precision |
| Context Recall | 0.683 | Retrieved context completeness |

> Golden set of 25 curated QA pairs (Portuguese/English). Contexts and expected answers were rewritten in April 2026 for richer grounding; scores are expected to increase on the next evaluation run. Run `python -m evaluation.ragas_eval` for latest results (`outputs/evaluation/ragas_results.json`).

### LLM-as-Judge

Independent LLM scores agent responses (1–5 scale):

| Criterion | Score |
|-----------|-------|
| Relevance | 5.0 |
| Factual Accuracy | 4.0 |
| Business Usefulness | 3.5 |

### LIME Explainability

- Per-feature contribution (Open, High, Low, Close, Volume)
- Time-step importance within the 60-day window
- Interactive visualization in the Evaluation tab

---

## 📊 Monitoring & Observability

| Tool | Purpose | Port |
|------|---------|------|
| **Prometheus** | Metrics collection & alerting | :9090 |
| **Grafana** | Dashboards & visualization | :3000 |
| **Evidently** | Data drift & model drift detection | — |
| **MLflow** | Experiment tracking & model registry | :5000 |
| **Optuna Dashboard** | HPO study visualization | :8080 |

### Telemetry Dashboard

Real-time health monitoring of all 6 services:

- ✅ / ❌ Status indicators with response time
- 🔗 Quick links (Swagger, ReDoc, GitHub, Targets, Dashboards)
- ℹ️ Expandable service info panels
- Auto-check on page load

### Business Metrics

- **P&L Tracking**: Cumulative profit/loss from prediction signals
- **SLA Monitoring**: Uptime, latency, error rate compliance
- **Cost Analysis**: Infrastructure + LLM token cost breakdown with daily trends

---

## 🐳 Docker

### Services

| Service | Port | Description | Compose File |
|---------|------|-------------|-------------|
| `api` | 8000 | FastAPI REST API | `docker-compose.api.yml` |
| `dashboard` | 3001 | Next.js frontend | `docker-compose.yml` |
| `mlflow` | 5000 | MLflow Tracking Server | `docker-compose.yml` |
| `prometheus` | 9090 | Metrics collection | `docker-compose.monitoring.yml` |
| `grafana` | 3000 | Monitoring dashboards | `docker-compose.monitoring.yml` |
| `nginx` | 80 | Reverse proxy / load balancer | `docker-compose.api.yml` |

### Commands

```bash
# Full stack
docker compose up -d

# API only (production)
docker compose -f docker-compose.api.yml up -d

# Scale API horizontally (3 replicas)
docker compose -f docker-compose.api.yml up -d --scale api=3

# Monitoring stack
docker compose -f docker-compose.monitoring.yml up -d

# Stop everything
docker compose down
```

---

## 🧪 Testing

**594 automated tests** across 46 test files with pytest:

```bash
make test
# or: pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test Modules

```bash
pytest tests/test_api/ -v          # API endpoints + schemas
pytest tests/test_models/ -v       # LSTM architecture
pytest tests/test_data/ -v         # Preprocessing & sequences
pytest tests/test_etl/ -v          # ETL extractors
pytest tests/test_training/ -v     # Training + Champion-Challenger
pytest tests/test_monitoring/ -v   # Drift + metrics + telemetry
pytest tests/test_security/ -v     # Guardrails + PII detection
pytest tests/test_agent/ -v        # ReAct agent + RAG pipeline
pytest tests/test_explainability/  # LIME + permutation importance
pytest tests/test_prediction/ -v   # Forecast + confidence intervals
pytest tests/test_utils/ -v        # Database manager + utilities
pytest tests/test_config/ -v       # Configuration validation
```

### Coverage Highlights

| Module | Coverage |
|--------|----------|
| `src/models/lstm_model.py` | 100% |
| `src/training/champion_challenger.py` | 99% |
| `src/training/train.py` | 97% |
| `src/security/guardrails.py` | 95% |
| `src/etl/preprocessing.py` | 93% |
| `src/prediction/predict.py` | 90% |

---

## ⚙️ CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

```
Push / PR → Lint → Format → Type Check → Security → Audit → Tests → Docker Build
```

| Step | Tool | Description |
|------|------|-------------|
| Lint | `ruff check` | Rules E, F, I, W |
| Format | `ruff format --check` | Consistent formatting |
| Type Check | `mypy` | Static type verification |
| Security | `bandit` | Static security analysis |
| Audit | `pip-audit` | Dependency vulnerabilities |
| Tests | `pytest --cov` | 594 tests, 60% minimum coverage |
| Docker | `docker build` | Image build validation |

---

## 🔒 Security

- ✅ Input / output guardrails for prompt injection protection
- ✅ PII detection and redaction
- ✅ OWASP LLM Top 10 mapping ([OWASP_MAPPING.md](docs/OWASP_MAPPING.md))
- ✅ Red Team testing report ([RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md))
- ✅ `bandit` for static security analysis
- ✅ `pip-audit` for dependency vulnerability scanning
- ✅ Dockerfile with non-root user (`appuser`)
- ✅ LGPD compliance plan ([LGPD_PLAN.md](docs/LGPD_PLAN.md))

See [SECURITY.md](SECURITY.md) for full details.

---

## 📁 Project Structure

```
nvidia-mlops-platform/
├── src/                              # Backend source code
│   ├── config.py                     #   Centralized configuration
│   ├── api/                          #   FastAPI REST API (66 endpoints)
│   │   ├── main.py                   #     App + lifespan
│   │   ├── schemas.py                #     Pydantic schemas
│   │   ├── dependencies.py           #     Dependency injection
│   │   └── routers/                  #     10 route modules
│   │       ├── health.py             #       Health + readiness
│   │       ├── data.py               #       Historical data
│   │       ├── predict.py            #       Prediction / inference
│   │       ├── train.py              #       Training (async + sync)
│   │       ├── agent.py              #       AI Agent chat
│   │       ├── evaluation_api.py     #       RAGAS + LLM-Judge + LIME
│   │       ├── mlops_api.py          #       Registry + P&L + Cost
│   │       ├── monitoring_api.py     #       Drift + metrics
│   │       ├── model_info.py         #       Model card info
│   │       └── logs.py              #       Application logs
│   ├── models/
│   │   └── lstm_model.py             #   LSTM architecture (NvidiaLSTM)
│   ├── training/
│   │   ├── train.py                  #   Training pipeline + MLflow
│   │   ├── hyperparameter_search.py  #   HPO with Optuna
│   │   ├── champion_challenger.py    #   Automated model promotion
│   │   └── model_registry.py         #   SQLite model registry
│   ├── prediction/
│   │   └── predict.py                #   Iterative forecast + visualization
│   ├── data/
│   │   └── preprocessing.py          #   Normalization, sequences, split
│   ├── etl/                          #   Data extraction & loading
│   ├── agent/                        #   ReAct agent + RAG + ChromaDB
│   ├── explainability/               #   LIME + permutation importance
│   ├── monitoring/                   #   Prometheus + Evidently + SLA
│   ├── security/                     #   Guardrails + PII detection
│   └── utils/                        #   Database manager + helpers
│
├── dashboard-frontend/               # Next.js 14 Dashboard
│   ├── src/app/                      #   11 pages (App Router)
│   │   ├── home/                     #     Overview
│   │   ├── predictions/              #     Forecast charts
│   │   ├── metrics/                  #     Model metrics
│   │   ├── evaluation/               #     RAGAS + LIME + LLM-Judge
│   │   ├── observability/            #     Drift + Telemetry
│   │   ├── mlops/                    #     Registry + P&L + Cost
│   │   ├── agent/                    #     AI Agent chat
│   │   ├── architecture/             #     System diagram
│   │   ├── model-schema/             #     Model card
│   │   ├── logs/                     #     Service logs
│   │   └── next-steps/               #     Roadmap (10 features)
│   ├── src/components/               #   Reusable UI components
│   ├── src/lib/api.ts                #   Typed API client
│   └── src/types/                    #   TypeScript interfaces
│
├── evaluation/                       # LLM Evaluation
│   ├── ragas_eval.py                 #   RAGAS + OpenRouter + HuggingFace
│   ├── llm_judge.py                  #   LLM-as-Judge scoring
│   └── ab_test_prompts.py            #   A/B prompt testing
│
├── tests/                            # 594 automated tests (46 files)
├── scripts/                          # Execution scripts
├── configs/                          # YAML configs + Prometheus + Grafana
├── data/                             # SQLite + golden set + processed data
├── notebooks/                        # Jupyter (EDA, evaluation, metrics)
├── docs/                             # Model Card, System Card, LGPD, OWASP
├── mlruns/                           # MLflow tracking data
│
├── docker-compose.yml                # Main services
├── docker-compose.api.yml            # Production API + Nginx
├── docker-compose.monitoring.yml     # Prometheus + Grafana
├── Dockerfile                        # Multi-stage image
├── Dockerfile.api                    # Optimized API image
├── Makefile                          # 15+ command shortcuts
├── pyproject.toml                    # Project metadata + tool configs
├── dvc.yaml                          # DVC pipeline definition
├── requirements.txt                  # Python dependencies
└── README.md                         # You are here
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Model card — architecture, training, limitations |
| [SYSTEM_CARD.md](docs/SYSTEM_CARD.md) | Full system documentation |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed experiment results and metrics |
| [LGPD_PLAN.md](docs/LGPD_PLAN.md) | LGPD / GDPR compliance plan |
| [OWASP_MAPPING.md](docs/OWASP_MAPPING.md) | OWASP LLM Top 10 security mapping |
| [RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) | Red team testing report |
| [LLM_BENCHMARK.md](docs/LLM_BENCHMARK.md) | LLM benchmark results |
| [SECURITY.md](SECURITY.md) | Security advisory and practices |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Step-by-step demo walkthrough |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Run validation: `make all` (lint + typecheck + security + tests)
4. Commit with [Conventional Commits](https://www.conventionalcommits.org/) (`git commit -m 'feat: add new feature'`)
5. Push and open a Pull Request

---

## 📄 License

This project is under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/LucasTechAI"><strong>LucasTechAI</strong></a>
  <br/>
  <a href="mailto:lucas.mendestech@gmail.com">lucas.mendestech@gmail.com</a>
</p>
