# 🟢 NVIDIA MLOps Platform

[![CI](https://github.com/LucasTechAI/nvidia-mlops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/LucasTechAI/nvidia-mlops-platform/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.5%2B-blue.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Plataforma MLOps end-to-end para previsão do preço das ações da NVIDIA usando **LSTM (Long Short-Term Memory)** com rastreamento de experimentos via **MLflow**, otimização de hiperparâmetros com **Optuna**, API REST com **FastAPI**, dashboard interativo com **Streamlit** e deploy containerizado com **Docker**.

> **FIAP Pós-Tech MLET** — Tech Challenge Fase 4 / Fase 5

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Requisitos](#-requisitos)
- [Instalação e Execução](#-instalação-e-execução)
- [Endpoints da API](#-endpoints-da-api)
- [Dashboard](#-dashboard)
- [Pipeline de Dados (ETL)](#-pipeline-de-dados-etl)
- [Treinamento do Modelo](#-treinamento-do-modelo)
- [Otimização de Hiperparâmetros](#-otimização-de-hiperparâmetros)
- [Métricas e Desempenho](#-métricas-e-desempenho)
- [MLflow Tracking](#-mlflow-tracking)
- [Docker](#-docker)
- [Testes](#-testes)
- [CI/CD](#-cicd)
- [Segurança](#-segurança)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

---

## 🎯 Visão Geral

O sistema prevê o preço de fechamento das ações da NVIDIA (NVDA) para os próximos **30 dias** usando dados históricos desde 2017. O pipeline completo inclui:

1. **ETL**: Extração de dados via Yahoo Finance → SQLite
2. **Treinamento**: LSTM em PyTorch com early stopping e gradient clipping
3. **HPO**: Busca bayesiana de hiperparâmetros com Optuna (50+ trials)
4. **Predição**: Forecast iterativo de 30 dias com intervalos de confiança
5. **API REST**: FastAPI com endpoints de treino, predição e dados
6. **Dashboard**: Streamlit com visualizações interativas
7. **Tracking**: MLflow para versionamento de modelos e métricas
8. **Deploy**: Docker Compose com múltiplos serviços

---

## 🏗 Arquitetura

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

## 📦 Requisitos

| Requisito | Versão |
|-----------|--------|
| Python | 3.12+ |
| PyTorch | ≥ 2.6.0 |
| CUDA (opcional) | 12.x |
| Docker + Compose | 24.x+ |
| RAM mínimo | 4 GB |
| Disco | ~2 GB (com deps) |

---

## 🚀 Instalação e Execução

### Opção 1: Execução Local (Recomendado para Desenvolvimento)

```bash
# 1. Clonar o repositório
git clone https://github.com/LucasTechAI/nvidia-mlops-platform.git
cd nvidia-mlops-platform

# 2. Criar ambiente virtual (opcional, recomendado)
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependências
make install
# ou: pip install -r requirements.txt

# 4. Configurar variáveis de ambiente (opcional)
cp .env.example .env
# Editar .env conforme necessário
```

#### Passo a passo completo de execução:

```bash
# ── ETAPA 1: Extrair dados ──────────────────────────────────
# Busca dados da NVIDIA via Yahoo Finance e carrega no SQLite
python3 scripts/run_etl_nvidia.py

# ── ETAPA 2: Treinar o modelo ───────────────────────────────
# Treina LSTM com 100 epochs, early stopping, MLflow tracking
bash scripts/run_training.sh

# ── ETAPA 3 (Opcional): Otimizar hiperparâmetros ────────────
# Executa 20 trials de otimização bayesiana com Optuna
bash scripts/run_hpo.sh 20

# ── ETAPA 4: Gerar previsões ────────────────────────────────
# Gera forecast de 30 dias (necessita run_id do MLflow)
bash scripts/run_prediction.sh <mlflow_run_id>

# ── ETAPA 5: Subir a API ────────────────────────────────────
# FastAPI em http://localhost:8000
bash scripts/run_api.sh

# ── ETAPA 6: Subir o Dashboard ──────────────────────────────
# Streamlit em http://localhost:8501
bash scripts/run_dashboard.sh

# ── ETAPA 7: Visualizar experimentos ────────────────────────
# MLflow UI em http://localhost:5000
bash scripts/start_mlflow_ui.sh
```

### Opção 2: Execução via Docker

```bash
# Build e start de todos os serviços
docker compose up -d

# Ou apenas a API (produção)
docker compose -f docker-compose.api.yml up -d

# Verificar status
docker compose ps

# Ver logs
docker compose logs -f api

# Parar tudo
docker compose down
```

### Opção 3: Via Makefile (atalhos)

```bash
make help          # Ver todos os comandos disponíveis
make data          # Rodar ETL
make train         # Treinar modelo
make hpo           # Otimizar hiperparâmetros
make serve         # Subir API
make dashboard     # Subir Dashboard
make mlflow-ui     # Subir MLflow UI
make test          # Rodar testes
make lint          # Verificar código
make all           # Lint + typecheck + security + testes
```

---

## 🌐 Endpoints da API

A API REST roda em **http://localhost:8000**. Documentação interativa disponível em `/docs` (Swagger UI).

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Health check (status, model_loaded, uptime) |
| `GET` | `/health/ready` | Readiness probe para orquestração |
| `GET` | `/data` | Retorna dados históricos da NVIDIA |
| `GET` | `/data/summary` | Estatísticas resumidas dos dados |
| `POST` | `/predict` | Gera previsão de N dias com intervalos de confiança |
| `POST` | `/predict/inference` | Inferência em sequência customizada |
| `POST` | `/train` | Inicia treinamento assíncrono (background) |
| `POST` | `/train/sync` | Treinamento síncrono (blocking) |
| `GET` | `/train/status` | Status do treinamento em andamento |
| `POST` | `/train/stop` | Solicitar parada do treinamento |

### Exemplos de uso

```bash
# Health check
curl http://localhost:8000/health

# Buscar dados históricos (últimos 30 dias)
curl "http://localhost:8000/data?limit=30"

# Gerar previsão de 30 dias
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon": 30, "confidence_level": 0.95}'

# Iniciar treinamento com parâmetros customizados
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"epochs": 50, "batch_size": 32, "learning_rate": 0.001}'

# Verificar status do treinamento
curl http://localhost:8000/train/status
```

---

## 📊 Dashboard

O dashboard Streamlit (**http://localhost:8501**) oferece:

- **Visão geral**: Preço atual, variação, volume
- **Gráfico de previsão**: Histórico + forecast com intervalos de confiança
- **Métricas do modelo**: RMSE, MAE, MAPE, loss curves
- **Análise de dados**: Estatísticas descritivas, distribuições
- **Arquitetura**: Diagrama visual da rede LSTM

```bash
bash scripts/run_dashboard.sh
# ou: make dashboard
```

---

## 🔄 Pipeline de Dados (ETL)

O pipeline ETL extrai dados da NVIDIA via Yahoo Finance e armazena em SQLite:

```
Yahoo Finance (NVDA) → CSV → SQLite (data/nvidia_stock.db)
```

- **Tabela**: `nvidia_stock` — 6.796+ registros
- **Colunas**: `date`, `open`, `high`, `low`, `close`, `volume`, `dividends`, `stock_splits`
- **Período**: Histórico completo disponível, filtrado a partir de 2017 para treino

```bash
python3 scripts/run_etl_nvidia.py
# ou: make data
```

---

## 🧠 Treinamento do Modelo

### Arquitetura LSTM

| Parâmetro | Valor Padrão |
|-----------|-------------|
| Tipo | Stacked LSTM (PyTorch `nn.LSTM`) |
| Camadas LSTM | 2 |
| Hidden Size | 128 |
| Dropout | 0.2 |
| Bidirecional | Não |
| Sequence Length | 60 dias (lookback window) |
| Output | 1 (preço de fechamento) |

### Configuração de Treino

| Parâmetro | Valor |
|-----------|-------|
| Epochs | 100 (com early stopping) |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Otimizador | Adam |
| Loss Function | MSE (Mean Squared Error) |
| Early Stopping | Patience = 10 epochs |
| Split | 70% treino / 15% validação / 15% teste |
| Normalização | MinMaxScaler (0-1) |

### Como treinar

```bash
bash scripts/run_training.sh
# ou: make train
```

O treinamento executa o seguinte pipeline:

1. Carrega dados do SQLite (a partir de 2017)
2. Normaliza features com MinMaxScaler
3. Cria sequências de 60 timesteps
4. Divide em train/val/test (70/15/15)
5. Treina com early stopping monitorando val_loss
6. Registra parâmetros, métricas e artefatos no MLflow
7. Salva modelo (`best_model.pth`) e scaler (`scaler.pkl`)

---

## 🔬 Otimização de Hiperparâmetros

Busca bayesiana via **Optuna** com TPE Sampler:

| Hiperparâmetro | Espaço de Busca |
|----------------|-----------------|
| `num_layers` | [1, 2, 3, 4] |
| `hidden_size` | [32, 64, 128, 256] |
| `learning_rate` | [1e-5, 1e-2] (escala log) |
| `dropout` | [0.1, 0.5] |
| `sequence_length` | [30, 60, 90, 120] |
| `batch_size` | [16, 32, 64, 128] |

**Objetivo**: Minimizar RMSE de validação

```bash
bash scripts/run_hpo.sh 20    # 20 trials
bash scripts/run_hpo.sh 50    # 50 trials (mais completo)
# ou: make hpo
```

---

## 📈 Métricas e Desempenho

O modelo é avaliado com as seguintes métricas:

| Métrica | Descrição |
|---------|-----------|
| **RMSE** | Root Mean Square Error — métrica principal de otimização |
| **MAE** | Mean Absolute Error — robusta a outliers |
| **MAPE** | Mean Absolute Percentage Error — erro relativo percentual |

### Modelo Treinado (Run `ee17873a`)

> Treinado em 02/02/2026 • PyTorch 2.10+CUDA • MLflow 3.8.1 • ~800 KB

| Configuração | Valor |
|-------------|-------|
| Arquitetura | 2-layer LSTM, hidden=128, dropout=0.2 |
| Features | OHLCV (Open, High, Low, Close, Volume) |
| Sequence Length | 60 dias |
| Epochs executados | 100 (com early stopping) |

**Artefatos gerados**:
- `loss_curves.png` — curvas de loss treino vs validação
- `predictions_vs_actual.png` — previsão vs valores reais no teste
- `scaler.joblib` — scaler para inverse transform

Os resultados detalhados de cada experimento estão em [EXPERIMENTS.md](EXPERIMENTS.md).

---

## 📦 MLflow Tracking

Todos os experimentos são rastreados automaticamente:

- **Parâmetros**: Arquitetura, hiperparâmetros, features utilizadas
- **Métricas**: Loss, RMSE, MAE, MAPE (por época)
- **Artefatos**: Modelo treinado, scaler, gráficos de loss e predição
- **Modelos**: Versionados via PyTorch flavor

```bash
bash scripts/start_mlflow_ui.sh
# Acesse http://localhost:5000
```

---

## 🐳 Docker

### Serviços disponíveis

| Serviço | Porta | Descrição | Compose File |
|---------|-------|-----------|-------------|
| `mlflow` | 5000 | MLflow Tracking Server | `docker-compose.yml` |
| `etl` | — | Pipeline de extração de dados | `docker-compose.yml` |
| `training` | — | Treinamento do modelo | `docker-compose.yml` (profile: training) |
| `dev` | — | Ambiente de desenvolvimento | `docker-compose.yml` (profile: dev) |
| `api` | 8000 | FastAPI (produção) | `docker-compose.api.yml` |
| `nginx` | 80 | Load balancer | `docker-compose.api.yml` (profile: production) |

### Comandos Docker

```bash
# Stack principal (MLflow + ETL)
docker compose up -d

# Apenas API (produção)
docker compose -f docker-compose.api.yml up -d

# API com escala horizontal (3 replicas)
docker compose -f docker-compose.api.yml up -d --scale api=3

# Treinamento via container
docker compose --profile training up training

# Ambiente de desenvolvimento
docker compose --profile dev run dev bash

# Parar tudo
docker compose down
```

---

## 🧪 Testes

**107 testes** automatizados com pytest:

```bash
# Rodar todos os testes
make test
# ou: pytest tests/ -v --cov=src --cov-report=term-missing

# Testes por módulo
pytest tests/test_api/ -v          # API endpoints + schemas
pytest tests/test_models/ -v       # Modelo LSTM
pytest tests/test_data/ -v         # Preprocessing
pytest tests/test_etl/ -v          # Extractor
```

### Cobertura por módulo

| Módulo | Cobertura |
|--------|-----------|
| `src/api/schemas.py` | 100% |
| `src/models/lstm_model.py` | 100% |
| `src/config.py` | 96% |
| `src/api/routers/health.py` | 90% |
| `src/api/routers/data.py` | 88% |
| `src/data/preprocessing.py` | 62% |

---

## ⚙️ CI/CD

Pipeline de integração contínua via **GitHub Actions** (`.github/workflows/ci.yml`):

```
Push/PR → Lint (ruff) → Format Check → Mypy → Bandit → pip-audit → Pytest → Docker Build
```

| Step | Ferramenta | Descrição |
|------|-----------|-----------|
| Lint | `ruff check` | Regras E, F, I, W |
| Format | `ruff format --check` | Formatação consistente |
| Type Check | `mypy` | Verificação de tipos |
| Security | `bandit` | Análise estática de segurança |
| Audit | `pip-audit` | Vulnerabilidades em dependências |
| Testes | `pytest --cov` | Cobertura mínima de 25% |
| Docker | `docker build` | Build de imagem (main/develop) |

---

## 🔒 Segurança

- ✅ Todas as dependências atualizadas (sem CVEs conhecidas)
- ✅ MLflow ≥ 3.5.0 (corrige DNS rebinding, RCE, desserialização)
- ✅ PyTorch ≥ 2.6.0 (corrige RCE e corrupção de memória)
- ✅ `bandit` para análise estática de segurança
- ✅ `pip-audit` para auditoria de dependências
- ✅ Dockerfile com usuário non-root (`appuser`)
- ✅ Volumes read-only onde possível

Veja [SECURITY.md](SECURITY.md) para detalhes completos.

---

## 📁 Estrutura do Projeto

```
nvidia-mlops-platform/
├── src/                              # Código-fonte principal
│   ├── config.py                     # Configuração centralizada (Settings dataclass)
│   ├── api/                          # FastAPI REST API
│   │   ├── main.py                   #   App FastAPI + lifespan
│   │   ├── schemas.py                #   Pydantic schemas (request/response)
│   │   ├── dependencies.py           #   Injeção de dependências (ModelState)
│   │   └── routers/                  #   Endpoints organizados por domínio
│   │       ├── health.py             #     Health check + readiness
│   │       ├── data.py               #     Dados históricos
│   │       ├── predict.py            #     Previsão / inferência
│   │       └── train.py              #     Treinamento (async + sync)
│   ├── models/
│   │   └── lstm_model.py             # Arquitetura LSTM (NvidiaLSTM nn.Module)
│   ├── training/
│   │   ├── train.py                  # Pipeline de treino com MLflow
│   │   └── hyperparameter_search.py  # HPO com Optuna
│   ├── prediction/
│   │   └── predict.py                # Forecast iterativo + visualização
│   ├── data/
│   │   └── preprocessing.py          # Normalização, sequências, split
│   ├── etl/
│   │   ├── extractor_nvidia.py       # Extração de dados via Yahoo Finance
│   │   ├── load_sqlite_nvidia.py     # Carga CSV → SQLite
│   │   └── preprocessing.py          # Preprocessing para ETL pipeline
│   ├── dashboard/
│   │   ├── app.py                    # Streamlit app
│   │   └── components/               # Componentes visuais do dashboard
│   ├── agent/                        # Agente ReAct com ferramentas (Fase 5)
│   ├── monitoring/                   # Prometheus, Evidently, Langfuse
│   ├── security/                     # Guardrails e detecção de PII
│   └── utils/
│       └── database_manager.py       # Gerenciador de conexão SQLite
├── tests/                            # 107 testes automatizados
├── evaluation/                       # Avaliação de LLM (RAGAS, LLM-as-judge)
├── scripts/                          # Scripts de execução (ETL, treino, API, etc.)
├── configs/                          # YAML configs (modelo, monitoramento, Prometheus)
├── data/                             # SQLite DB + golden_set + raw/processed
├── notebooks/                        # Jupyter notebooks (EDA, avaliação, métricas)
├── mlruns/                           # MLflow tracking data e artefatos
├── Dockerfile                        # Imagem Docker multi-stage
├── Dockerfile.api                    # Imagem otimizada para API
├── docker-compose.yml                # Serviços principais
├── docker-compose.api.yml            # Stack de produção (API + Nginx)
├── docker-compose.monitoring.yml     # Stack de monitoramento
├── Makefile                          # Atalhos de comandos
├── pyproject.toml                    # Metadata do projeto + configs de ferramentas
├── requirements.txt                  # Dependências Python
├── .github/workflows/ci.yml          # Pipeline CI/CD
├── EXPERIMENTS.md                    # Documentação de experimentos e métricas
└── SECURITY.md                       # Advisory de segurança
```

---

## 🤝 Contribuição

1. Fork o repositório
2. Crie uma feature branch (`git checkout -b feature/nova-feature`)
3. Faça as alterações e rode `make all` para validar
4. Commit com [Conventional Commits](https://www.conventionalcommits.org/) (`git commit -m 'feat: adicionar nova feature'`)
5. Push (`git push origin feature/nova-feature`)
6. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT — veja [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

- **LucasTechAI** — [@LucasTechAI](https://github.com/LucasTechAI) — lucas.mendestech@gmail.com
