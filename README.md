# 🟢 NVIDIA MLOps Platform

[![CI](https://github.com/LucasTechAI/nvidia-mlops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/LucasTechAI/nvidia-mlops-platform/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.5%2B-blue.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end MLOps platform for NVIDIA stock price prediction using **LSTM (Long Short-Term Memory)** with experiment tracking via **MLflow**, hyperparameter optimization with **Optuna**, REST API with **FastAPI**, interactive dashboard with **Streamlit**, and containerized deployment with **Docker**.

> **FIAP Post-Tech MLET** — Tech Challenge Phase 4 / Phase 5

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Requirements](#-requirements)
- [Installation & Running](#-installation--running)
- [API Endpoints](#-api-endpoints)
- [Dashboard](#-dashboard)
- [Data Pipeline (ETL)](#-data-pipeline-etl)
- [Model Training](#-model-training)
- [Hyperparameter Optimization](#-hyperparameter-optimization)
- [Metrics & Performance](#-metrics--performance)
- [MLflow Tracking](#-mlflow-tracking)
- [Docker](#-docker)
- [Testing](#-testing)
- [CI/CD](#-cicd)
- [Security](#-security)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The system predicts the closing price of NVIDIA (NVDA) stock for the next **30 days** using historical data since 2017. The complete pipeline includes:

1. **ETL**: Data extraction via Yahoo Finance → SQLite
2. **Training**: LSTM in PyTorch with early stopping and gradient clipping
3. **HPO**: Bayesian hyperparameter search with Optuna (50+ trials)
4. **Prediction**: Iterative 30-day forecast with confidence intervals
5. **REST API**: FastAPI with training, prediction, and data endpoints
6. **Dashboard**: Streamlit with interactive visualizations
7. **Tracking**: MLflow for model and metric versioning
8. **Deployment**: Docker Compose with multiple services

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NVIDIA MLOps Platform                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Yahoo    │───▶│ ETL Pipeline │───▶│ SQLite DB    │              │
│  │ Finance  │    │ (yfinance)   │    │ (6700+ rows) │              │
│  └──────────┘    └──────────────┘    └──────┬───────┘              │
│                                             │                       │
│                  ┌──────────────────────────┼───────────┐           │
│                  │       Data Pipeline      │           │           │
│                  │                          ▼           │           │
│                  │  ┌──────────────┐  ┌──────────┐     │           │
│                  │  │ Preprocessing│  │ Sequence │     │           │
│                  │  │ MinMaxScaler │─▶│ Generator│     │           │
│                  │  └──────────────┘  └────┬─────┘     │           │
│                  │                         │           │           │
│                  │    ┌────────────────────┼────┐      │           │
│                  │    │    LSTM Model       │    │      │           │
│                  │    │  ┌──────────────┐  │    │      │           │
│                  │    │  │ 2x LSTM Layer│  │    │      │           │
│                  │    │  │ Hidden: 128  │  │    │      │           │
│                  │    │  │ Dropout: 0.2 │  │    │      │           │
│                  │    │  └──────┬───────┘  │    │      │           │
│                  │    │         ▼          │    │      │           │
│                  │    │  ┌──────────────┐  │    │      │           │
│                  │    │  │ Dense Layer  │  │    │      │           │
│                  │    │  │ → Forecast   │  │    │      │           │
│                  │    │  └──────────────┘  │    │      │           │
│                  │    └───────────────────────┘  │      │           │
│                  │         Training Pipeline      │      │           │
│                  └──────────────────────────────────┘    │           │
│                                                         │           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │           │
│  │  FastAPI     │  │  Streamlit   │  │   MLflow     │  │           │
│  │  REST API    │  │  Dashboard   │  │   Tracking   │  │           │
│  │  :8000       │  │  :8501       │  │   :5000      │  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘  │           │
│                                                         │           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| PyTorch | ≥ 2.6.0 |
| CUDA (optional) | 12.x |
| Docker + Compose | 24.x+ |
| Minimum RAM | 4 GB |
| Disk | ~2 GB (with deps) |

---

## 🚀 Installation & Running

### Option 1: Local Execution (Recommended for Development)

```bash
# 1. Clone the repository
git clone https://github.com/LucasTechAI/nvidia-mlops-platform.git
cd nvidia-mlops-platform

# 2. Create virtual environment (optional, recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
make install
# or: pip install -r requirements.txt

# 4. Configure environment variables (optional)
cp .env.example .env
# Edit .env as needed
```

#### Step-by-step full execution:

```bash
# ── STEP 1: Extract data ────────────────────────────────────
# Fetches NVIDIA data via Yahoo Finance and loads into SQLite
python3 scripts/run_etl_nvidia.py

# ── STEP 2: Train the model ─────────────────────────────────
# Trains LSTM with 100 epochs, early stopping, MLflow tracking
bash scripts/run_training.sh

# ── STEP 3 (Optional): Optimize hyperparameters ─────────────
# Runs 20 trials of Bayesian optimization with Optuna
bash scripts/run_hpo.sh 20

# ── STEP 4: Generate predictions ────────────────────────────
# Generates 30-day forecast (requires MLflow run_id)
bash scripts/run_prediction.sh <mlflow_run_id>

# ── STEP 5: Start the API ───────────────────────────────────
# FastAPI at http://localhost:8000
bash scripts/run_api.sh

# ── STEP 6: Start the Dashboard ─────────────────────────────
# Streamlit at http://localhost:8501
bash scripts/run_dashboard.sh

# ── STEP 7: View experiments ────────────────────────────────
# MLflow UI at http://localhost:5000
bash scripts/start_mlflow_ui.sh
```

### Option 2: Docker Execution

```bash
# Build and start all services
docker compose up -d

# Or just the API (production)
docker compose -f docker-compose.api.yml up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api

# Stop everything
docker compose down
```

### Option 3: Via Makefile (shortcuts)

```bash
make help          # List all available commands
make data          # Run ETL
make train         # Train model
make hpo           # Optimize hyperparameters
make serve         # Start API
make dashboard     # Start Dashboard
make mlflow-ui     # Start MLflow UI
make test          # Run tests
make lint          # Check code
make all           # Lint + typecheck + security + tests
```

---

## 🌐 API Endpoints

The REST API runs at **http://localhost:8000**. Interactive documentation available at `/docs` (Swagger UI).

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Health check (status, model_loaded, uptime) |
| `GET` | `/health/ready` | Readiness probe for orchestration |
| `GET` | `/data` | Returns NVIDIA historical data |
| `GET` | `/data/summary` | Summary statistics of the data |
| `POST` | `/predict` | Generate N-day forecast with confidence intervals |
| `POST` | `/predict/inference` | Inference on custom sequence |
| `POST` | `/train` | Start asynchronous training (background) |
| `POST` | `/train/sync` | Synchronous training (blocking) |
| `GET` | `/train/status` | Status of ongoing training |
| `POST` | `/train/stop` | Request training stop |

### Usage Examples

```bash
# Health check
curl http://localhost:8000/health

# Fetch historical data (last 30 days)
curl "http://localhost:8000/data?limit=30"

# Generate 30-day forecast
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 30, "confidence_level": 0.95}'

# Start training with custom parameters
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 32, "learning_rate": 0.001}'

# Check training status
curl http://localhost:8000/train/status
```

---

## 📊 Dashboard

The Streamlit dashboard (**http://localhost:8501**) provides:

- **Overview**: Current price, change, volume
- **Forecast chart**: Historical data + forecast with confidence intervals
- **Model metrics**: RMSE, MAE, MAPE, loss curves
- **Data analysis**: Descriptive statistics, distributions
- **Architecture**: Visual diagram of the LSTM network
- **Observability**: Drift detection, champion-challenger status, telemetry
- **Evaluation**: Model evaluation metrics, explainability, LLM-judge results
- **AI Agent**: Interactive chat interface with RAG agent

```bash
bash scripts/run_dashboard.sh
# or: make dashboard
```

---

## 🔄 Data Pipeline (ETL)

The ETL pipeline extracts NVIDIA data via Yahoo Finance and stores it in SQLite:

```
Yahoo Finance (NVDA) → CSV → SQLite (data/nvidia_stock.db)
```

- **Table**: `nvidia_stock` — 6,796+ records
- **Columns**: `date`, `open`, `high`, `low`, `close`, `volume`, `dividends`, `stock_splits`
- **Period**: Full available history, filtered from 2017 for training

```bash
python3 scripts/run_etl_nvidia.py
# or: make data
```

---

## 🧠 Model Training

### LSTM Architecture

| Parameter | Default Value |
|-----------|--------------|
| Type | Stacked LSTM (PyTorch `nn.LSTM`) |
| LSTM Layers | 2 |
| Hidden Size | 128 |
| Dropout | 0.2 |
| Bidirectional | No |
| Sequence Length | 60 days (lookback window) |
| Output | 1 (closing price) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 (with early stopping) |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE (Mean Squared Error) |
| Early Stopping | Patience = 10 epochs |
| Split | 70% train / 15% validation / 15% test |
| Normalization | MinMaxScaler (0–1) |

### How to Train

```bash
bash scripts/run_training.sh
# or: make train
```

The training pipeline executes the following steps:

1. Loads data from SQLite (from 2017 onwards)
2. Normalizes features with MinMaxScaler
3. Creates sequences of 60 timesteps
4. Splits into train/val/test (70/15/15)
5. Trains with early stopping monitoring val_loss
6. Logs parameters, metrics, and artifacts to MLflow
7. Saves model (`best_model.pth`) and scaler (`scaler.pkl`)

---

## 🔬 Hyperparameter Optimization

Bayesian search via **Optuna** with TPE Sampler:

| Hyperparameter | Search Space |
|----------------|-------------|
| `num_layers` | [1, 2, 3, 4] |
| `hidden_size` | [32, 64, 128, 256] |
| `learning_rate` | [1e-5, 1e-2] (log scale) |
| `dropout` | [0.1, 0.5] |
| `sequence_length` | [30, 60, 90, 120] |
| `batch_size` | [16, 32, 64, 128] |

**Objective**: Minimize validation RMSE

```bash
bash scripts/run_hpo.sh 20    # 20 trials
bash scripts/run_hpo.sh 50    # 50 trials (more thorough)
# or: make hpo
```

---

## 📈 Metrics & Performance

The model is evaluated with the following metrics:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error — primary optimization metric |
| **MAE** | Mean Absolute Error — robust to outliers |
| **MAPE** | Mean Absolute Percentage Error — relative percentage error |

### Trained Model (Run `ee17873a`)

> Trained on 2026-02-02 • PyTorch 2.10+CUDA • MLflow 3.8.1 • ~800 KB

| Metric | Validation | Test |
|--------|-----------|------|
| **Loss (MSE)** | 0.003162 | 0.019251 |
| **RMSE** | 0.053320 | 0.137608 |
| **MAE** | 0.030938 | 0.080397 |

> Metrics on normalized scale (0–1). Dataset: 2,319 records (2017–2026), split 70/15/15.

| Configuration | Value |
|--------------|-------|
| Architecture | 2-layer LSTM, hidden=128, dropout=0.2 |
| Features | OHLCV (Open, High, Low, Close, Volume) |
| Sequence Length | 60 days |
| Epochs executed | 100 (with early stopping) |

**Generated artifacts**:
- `loss_curves.png` — training loss vs validation loss curves
- `predictions_vs_actual.png` — prediction vs actual values on test set
- `scaler.joblib` — scaler for inverse transform

Detailed results of each experiment are in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 📦 MLflow Tracking

All experiments are automatically tracked:

- **Parameters**: Architecture, hyperparameters, features used
- **Metrics**: Loss, RMSE, MAE, MAPE (per epoch)
- **Artifacts**: Trained model, scaler, loss and prediction plots
- **Models**: Versioned via PyTorch flavor

```bash
bash scripts/start_mlflow_ui.sh
# Go to http://localhost:5000
```

---

## 🐳 Docker

### Available Services

| Service | Port | Description | Compose File |
|---------|------|-------------|-------------|
| `mlflow` | 5000 | MLflow Tracking Server | `docker-compose.yml` |
| `etl` | — | Data extraction pipeline | `docker-compose.yml` |
| `training` | — | Model training | `docker-compose.yml` (profile: training) |
| `dev` | — | Development environment | `docker-compose.yml` (profile: dev) |
| `api` | 8000 | FastAPI (production) | `docker-compose.api.yml` |
| `nginx` | 80 | Load balancer | `docker-compose.api.yml` (profile: production) |

### Docker Commands

```bash
# Main stack (MLflow + ETL)
docker compose up -d

# API only (production)
docker compose -f docker-compose.api.yml up -d

# API with horizontal scaling (3 replicas)
docker compose -f docker-compose.api.yml up -d --scale api=3

# Training via container
docker compose --profile training up training

# Development environment
docker compose --profile dev run dev bash

# Stop everything
docker compose down
```

---

## 🧪 Testing

**484 automated tests** with pytest:

```bash
# Run all tests
make test
# or: pytest tests/ -v --cov=src --cov-report=term-missing

# Tests by module
pytest tests/test_api/ -v          # API endpoints + schemas
pytest tests/test_models/ -v       # LSTM model
pytest tests/test_data/ -v         # Preprocessing
pytest tests/test_etl/ -v          # Extractor
pytest tests/test_training/ -v     # Training + champion-challenger
pytest tests/test_monitoring/ -v   # Drift + metrics + telemetry
pytest tests/test_security/ -v     # Guardrails + PII detection
pytest tests/test_agent/ -v        # ReAct agent + RAG pipeline
```

### Coverage by Module

| Module | Coverage |
|--------|----------|
| `src/models/lstm_model.py` | 100% |
| `src/training/champion_challenger.py` | 99% |
| `src/training/train.py` | 97% |
| `src/security/guardrails.py` | 95% |
| `src/etl/preprocessing.py` | 93% |
| `src/prediction/predict.py` | 90% |
| **Total** | **61%** |

---

## ⚙️ CI/CD

Continuous integration pipeline via **GitHub Actions** (`.github/workflows/ci.yml`):

```
Push/PR → Lint (ruff) → Format Check → Mypy → Bandit → pip-audit → Pytest → Docker Build
```

| Step | Tool | Description |
|------|------|-------------|
| Lint | `ruff check` | Rules E, F, I, W |
| Format | `ruff format --check` | Consistent formatting |
| Type Check | `mypy` | Type verification |
| Security | `bandit` | Static security analysis |
| Audit | `pip-audit` | Dependency vulnerabilities |
| Tests | `pytest --cov` | Minimum coverage of 60% |
| Docker | `docker build` | Image build (main/develop) |

---

## 🔒 Security

- ✅ All dependencies up to date (no known CVEs)
- ✅ MLflow ≥ 3.5.0 (fixes DNS rebinding, RCE, deserialization)
- ✅ PyTorch ≥ 2.6.0 (fixes RCE and memory corruption)
- ✅ `bandit` for static security analysis
- ✅ `pip-audit` for dependency auditing
- ✅ Dockerfile with non-root user (`appuser`)
- ✅ Read-only volumes where possible
- ✅ Input/output guardrails for prompt injection and PII detection
- ✅ OWASP LLM Top 10 mapping

See [SECURITY.md](SECURITY.md) for full details.

---

## 📁 Project Structure

```
nvidia-mlops-platform/
├── src/                              # Main source code
│   ├── config.py                     # Centralized configuration (Settings dataclass)
│   ├── api/                          # FastAPI REST API
│   │   ├── main.py                   #   FastAPI app + lifespan
│   │   ├── schemas.py                #   Pydantic schemas (request/response)
│   │   ├── dependencies.py           #   Dependency injection (ModelState)
│   │   └── routers/                  #   Endpoints organized by domain
│   │       ├── health.py             #     Health check + readiness
│   │       ├── data.py               #     Historical data
│   │       ├── predict.py            #     Prediction / inference
│   │       └── train.py              #     Training (async + sync)
│   ├── models/
│   │   └── lstm_model.py             # LSTM architecture (NvidiaLSTM nn.Module)
│   ├── training/
│   │   ├── train.py                  # Training pipeline with MLflow
│   │   ├── hyperparameter_search.py  # HPO with Optuna
│   │   └── champion_challenger.py    # Automated model promotion
│   ├── prediction/
│   │   └── predict.py                # Iterative forecast + visualization
│   ├── data/
│   │   └── preprocessing.py          # Normalization, sequences, split
│   ├── etl/
│   │   ├── extractor_nvidia.py       # Data extraction via Yahoo Finance
│   │   ├── load_sqlite_nvidia.py     # CSV → SQLite loading
│   │   └── preprocessing.py          # Preprocessing for ETL pipeline
│   ├── explainability/
│   │   └── feature_importance.py     # Permutation importance + MLflow logging
│   ├── dashboard/
│   │   ├── app.py                    # Streamlit app
│   │   └── components/               # Dashboard visual components
│   ├── agent/                        # ReAct agent with tools (Phase 5)
│   ├── monitoring/                   # Prometheus, Evidently, Langfuse
│   ├── security/                     # Guardrails and PII detection
│   └── utils/
│       └── database_manager.py       # SQLite connection manager
├── tests/                            # 484 automated tests
├── evaluation/                       # LLM evaluation (RAGAS, LLM-as-judge)
├── scripts/                          # Execution scripts (ETL, training, API, etc.)
├── configs/                          # YAML configs (model, monitoring, Prometheus, Grafana)
├── data/                             # SQLite DB + golden_set + raw/processed
├── notebooks/                        # Jupyter notebooks (EDA, evaluation, metrics)
├── mlruns/                           # MLflow tracking data and artifacts
├── docs/                             # Documentation (Model Card, System Card, LGPD, OWASP, etc.)
├── Dockerfile                        # Multi-stage Docker image
├── Dockerfile.api                    # Optimized API image
├── docker-compose.yml                # Main services
├── docker-compose.api.yml            # Production stack (API + Nginx)
├── docker-compose.monitoring.yml     # Monitoring stack
├── Makefile                          # Command shortcuts
├── pyproject.toml                    # Project metadata + tool configs
├── requirements.txt                  # Python dependencies
├── .github/workflows/ci.yml          # CI/CD pipeline
├── EXPERIMENTS.md                    # Experiment documentation and metrics
└── SECURITY.md                       # Security advisory
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and run `make all` to validate
4. Commit with [Conventional Commits](https://www.conventionalcommits.org/) (`git commit -m 'feat: add new feature'`)
5. Push (`git push origin feature/new-feature`)
6. Open a Pull Request

---

## 📄 License

This project is under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

- **LucasTechAI** — [@LucasTechAI](https://github.com/LucasTechAI) — lucas.mendestech@gmail.com
