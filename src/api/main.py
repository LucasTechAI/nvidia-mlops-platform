"""
FastAPI main application.

NVIDIA Stock Price Prediction API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import model_state
from src.api.routers import (
    agent_router,
    data_router,
    evaluation_router,
    health_router,
    logs_router,
    mlops_router,
    model_router,
    monitoring_router,
    predict_router,
    train_router,
)
from src.config import enable_mlflow_tracing
from src.monitoring.metrics import ACTIVE_REQUESTS, get_metrics, track_request

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Install structured log handler (SQLite)
try:
    from src.utils.log_database import LogDatabase, install_log_handler, seed_sample_logs

    install_log_handler(level=logging.INFO)
    # Seed sample data if DB is empty
    db = LogDatabase.get_instance()
    stats = db.get_stats(since_minutes=999_999)
    if stats.total == 0:
        seed_sample_logs()
except Exception as _exc:
    logger.warning("Could not install SQLite log handler: %s", _exc)

# Seed MLOps demo data (business metrics, SLA, feature store, registry, canary)
try:
    from src.monitoring.business_metrics import BusinessMetricsTracker

    BusinessMetricsTracker.get_instance().seed_sample_data()
except Exception as _exc:
    logger.warning("Could not seed business metrics: %s", _exc)

try:
    from src.monitoring.sla_monitor import SLAMonitor

    SLAMonitor.get_instance().seed_sample_data()
except Exception as _exc:
    logger.warning("Could not seed SLA data: %s", _exc)

try:
    from src.data.feature_store import FeatureStore

    FeatureStore.get_instance().seed_sample_data()
except Exception as _exc:
    logger.warning("Could not seed feature store: %s", _exc)

try:
    from src.training.model_registry import ModelRegistry

    ModelRegistry.get_instance().seed_sample_data()
except Exception as _exc:
    logger.warning("Could not seed model registry: %s", _exc)

try:
    from src.monitoring.canary_deploy import CanaryDeployManager

    CanaryDeployManager.get_instance().seed_sample_data()
except Exception as _exc:
    logger.warning("Could not seed canary deploy: %s", _exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NVIDIA Stock Prediction API...")

    # Enable MLflow tracing for LLM calls
    enable_mlflow_tracing()
    logger.info("MLflow tracing enabled")

    # Load model on startup
    success = model_state.load_model()
    if success:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - some endpoints may not work")

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="NVIDIA Stock Prediction API",
    description="""REST API for NVIDIA stock price prediction using LSTM neural networks.

## Features

- **Prediction**: Generate stock price forecasts with confidence intervals
- **Inference**: Run predictions on custom input sequences
- **Training**: Train or retrain the LSTM model
- **Data**: Access historical stock data
- **Health**: Monitor API and model status

## Model

Uses a bidirectional LSTM trained on NVIDIA historical stock prices
with Monte Carlo Dropout for uncertainty estimation.

## Usage

1. Check API health: `GET /health`
2. Get predictions: `POST /predict`
3. Access data: `GET /data/historical`
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(train_router)
app.include_router(data_router)
app.include_router(agent_router)
app.include_router(model_router)
app.include_router(monitoring_router)
app.include_router(evaluation_router)
app.include_router(logs_router)
app.include_router(mlops_router)


# Prometheus metrics middleware
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Track request metrics for Prometheus and SLA monitoring."""
    import time

    method = request.method
    endpoint = request.url.path

    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        track_request(method, endpoint, response.status_code, duration)

        # Record in SLA monitor
        try:
            from src.monitoring.sla_monitor import SLAMonitor

            SLAMonitor.get_instance().record_request(method, endpoint, response.status_code, duration * 1000)
        except Exception:
            pass

        return response
    except Exception:
        duration = time.time() - start_time
        track_request(method, endpoint, 500, duration)
        raise
    finally:
        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type="text/plain; charset=utf-8")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NVIDIA Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "inference": "/predict/inference",
            "train": "/train",
            "data": "/data",
            "agent": "/agent/query",
            "health": "/health",
        },
    }


# For running with: python -m api.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
