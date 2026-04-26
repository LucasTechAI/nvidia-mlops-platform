# System Card — NVIDIA MLOps Platform

## System Overview

| Field | Value |
|-------|-------|
| **System Name** | NVIDIA MLOps Stock Prediction Platform |
| **Version** | 1.0.0 |
| **Purpose** | End-to-end ML platform for NVIDIA stock price prediction and analysis |
| **Developed by** | Lucas (FIAP Post-Tech MLET — Tech Challenge Fase 5) |
| **Stack** | Python, PyTorch, FastAPI, Next.js 14, MLflow, Docker |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Next.js 14  │───▶│  FastAPI      │───▶│  LSTM Model      │
│  Dashboard   │    │  REST API     │    │  (PyTorch)       │
│  (3001)      │    │  (8000)       │    └──────────────────┘
└─────────────┘    │               │    ┌──────────────────┐
                   │  /predict     │───▶│  SQLite DB       │
                   │  /train       │    │  (Stock Data)    │
                   │  /agent/query │    └──────────────────┘
                   │  /data        │    ┌──────────────────┐
                   │  /health      │───▶│  ReAct Agent     │
                   └──────────────┘    │  + RAG Pipeline   │
                                       │  + LLM (OpenRouter│
┌─────────────┐    ┌──────────────┐    │   Gemini 2.0)    │
│  Prometheus  │───▶│  Grafana     │    └──────────────────┘
│  (9090)      │    │  (3000)      │
└─────────────┘    └──────────────┘    ┌──────────────────┐
                                       │  MLflow          │
                                       │  (5000)          │
                                       └──────────────────┘
```

## Components

### 1. Data Pipeline (ETL)
- **Source**: Yahoo Finance API (yfinance)
- **Storage**: SQLite database
- **Processing**: MinMaxScaler normalization, sequence windowing
- **Schedule**: On-demand via API or scripts

### 2. LSTM Model
- **Architecture**: 2-layer LSTM, 128 hidden units, 5 features
- **Training**: Adam optimizer, MSE loss, early stopping
- **Inference**: Monte Carlo Dropout for uncertainty
- **Tracking**: MLflow experiment tracking

### 3. ReAct Agent (LLM)
- **Pattern**: ReAct (Reasoning + Acting) loop
- **Tools**: Stock data query, LSTM prediction, model metrics, RAG search
- **RAG**: ChromaDB vector store with financial domain knowledge
- **Providers**: OpenAI or Groq (configurable)

### 4. REST API (FastAPI)
- **Endpoints**: /health, /predict, /train, /data, /agent/query, /evaluation/*, /mlops/*, /monitoring/*
- **Docs**: Swagger UI (/docs), ReDoc (/redoc)
- **Auth**: CORS configured (production-ready)

### 5. Dashboard (Next.js 14)
- **Framework**: Next.js 14 (App Router) + TypeScript + Tailwind CSS
- **Pages**: 11 pages — Home, Predictions, Metrics, Evaluation, Observability, MLOps, AI Agent, Architecture, Model Schema, Logs, Next Steps
- **URL**: http://localhost:3001
- **Charts**: Recharts (Line, Bar, Area, Pie, Radar)

### 6. Monitoring
- **Prometheus**: HTTP metrics, prediction latency, agent metrics
- **Evidently**: Data drift detection (PSI)
- **Langfuse**: LLM telemetry and tracing

### 7. Security
- **Input guardrails**: Prompt injection detection, topic validation
- **Output guardrails**: PII detection (Presidio), content filtering, disclaimers
- **OWASP**: LLM Top 10 compliance

## Data Flow

1. **Ingestion**: Yahoo Finance → ETL → SQLite
2. **Training**: SQLite → Preprocessing → LSTM Training → MLflow → Model Checkpoint
3. **Prediction**: API Request → Load Model → Generate Forecast → Return with CI
4. **Agent**: User Query → Guardrails → ReAct Loop → Tools → LLM → Guardrails → Response

## Deployment

### Docker Compose Services
- `training`: Model training container
- `api` + `nginx`: Production API with reverse proxy
- `prometheus` + `grafana`: Monitoring stack

### Environment Variables
| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | openrouter (default) |
| `LLM_MODEL` | Model name (e.g., google/gemini-2.0-flash-001) |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENROUTER_BASE_URL` | https://openrouter.ai/api/v1 |
| `MLFLOW_TRACKING_URI` | SQLite tracking URI |
| `DATABASE_PATH` | SQLite database path |

## Limitations

- Single-stock focus (NVIDIA only)
- Dependent on external LLM providers for agent functionality
- SQLite limits concurrent write access
- No real-time streaming data
- Model accuracy degrades for horizons > 30 days

## Responsible AI

- All predictions include risk disclaimers
- PII is detected and removed from inputs/outputs
- Prompt injection attacks are blocked
- Topic is restricted to financial domain
- See [MODEL_CARD.md](MODEL_CARD.md) for model-specific risks
- See [LGPD_PLAN.md](LGPD_PLAN.md) for data privacy compliance
- See [OWASP_MAPPING.md](OWASP_MAPPING.md) for security mapping
