#!/usr/bin/env bash
# =============================================================================
# NVIDIA MLOps Platform — Run All Services
# =============================================================================
# This script builds and starts every component of the platform via Docker.
# Usage:
#   bash scripts/run_all.sh          # Full pipeline
#   bash scripts/run_all.sh --skip-build   # Skip Docker build (faster)
#   bash scripts/run_all.sh --skip-train   # Skip training (use existing model)
# =============================================================================

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_BUILD=false
SKIP_TRAIN=false
SKIP_MONITORING=false

for arg in "$@"; do
    case $arg in
        --skip-build)      SKIP_BUILD=true ;;
        --skip-train)      SKIP_TRAIN=true ;;
        --skip-monitoring) SKIP_MONITORING=true ;;
        --help|-h)
            echo "Usage: bash scripts/run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-build       Skip Docker image build"
            echo "  --skip-train       Skip model training (use existing model)"
            echo "  --skip-monitoring  Skip Prometheus + Grafana"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
    esac
done

# ─── Helper functions ────────────────────────────────────────────────────────
step_count=0

step() {
    step_count=$((step_count + 1))
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  STEP ${step_count}: $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

success() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}  ⚠️  $1${NC}"
}

fail() {
    echo -e "${RED}  ❌ $1${NC}"
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=0

    echo -n "  Waiting for $name "
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo ""
            success "$name is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo ""
    warn "$name did not respond after ${max_attempts} attempts (may still be starting)"
    return 1
}

cleanup_on_exit() {
    if [ $? -ne 0 ]; then
        echo ""
        fail "Script failed. Services may be partially running."
        echo -e "  Run ${YELLOW}bash scripts/run_all.sh --skip-build${NC} to retry without rebuilding."
        echo -e "  Run ${YELLOW}docker compose down${NC} to stop everything."
    fi
}
trap cleanup_on_exit EXIT

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║                                                           ║"
echo "  ║        🟢  NVIDIA MLOps Platform — Full Setup  🟢        ║"
echo "  ║                                                           ║"
echo "  ║   ETL → Training → API → Dashboard → Monitoring → Tests  ║"
echo "  ║                                                           ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ${CYAN}Project root:${NC}  $PROJECT_ROOT"
echo -e "  ${CYAN}Skip build:${NC}    $SKIP_BUILD"
echo -e "  ${CYAN}Skip train:${NC}    $SKIP_TRAIN"
echo -e "  ${CYAN}Skip monitor:${NC}  $SKIP_MONITORING"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Ensure prerequisites
# ═════════════════════════════════════════════════════════════════════════════
step "Checking prerequisites"

if ! command -v docker &> /dev/null; then
    fail "Docker is not installed. Please install Docker first."
    exit 1
fi
success "Docker found: $(docker --version | head -1)"

if ! docker compose version &> /dev/null 2>&1; then
    fail "Docker Compose (v2) is not available."
    exit 1
fi
success "Docker Compose found: $(docker compose version --short 2>/dev/null || echo 'v2')"

if ! docker info &> /dev/null 2>&1; then
    fail "Docker daemon is not running. Start Docker first."
    exit 1
fi
success "Docker daemon is running"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Create required directories
# ═════════════════════════════════════════════════════════════════════════════
step "Creating required directories"

mkdir -p data/raw data/processed data/models/checkpoints data/outputs data/mlruns
mkdir -p logs outputs mlruns models
success "All directories created"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Build Docker images
# ═════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_BUILD" = false ]; then
    step "Building Docker images"
    echo "  Building main image (Dockerfile)..."
    docker compose build --no-cache 2>&1 | tail -5
    success "Main image built"

    echo "  Building API image (Dockerfile.api)..."
    docker compose -f docker-compose.api.yml build 2>&1 | tail -5
    success "API image built"
else
    step "Skipping Docker build (--skip-build)"
    warn "Using existing images"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Run ETL Pipeline
# ═════════════════════════════════════════════════════════════════════════════
step "Running ETL Pipeline (Yahoo Finance → SQLite)"

# Run ETL with the dev profile container (has full source code)
docker compose run --rm -e PYTHONPATH=/app dev \
    python scripts/run_etl_nvidia.py 2>&1 | tail -15

if [ -f "data/nvidia_stock.db" ] || [ -f "data/raw/nvidia_stock.csv" ]; then
    success "ETL completed — data/nvidia_stock.db ready"
    if [ -f "data/raw/nvidia_stock.csv" ]; then
        ROW_COUNT=$(wc -l < data/raw/nvidia_stock.csv)
        success "CSV has $ROW_COUNT rows"
    fi
else
    # Try running locally if Docker dev profile fails
    warn "Docker ETL had issues, trying local execution..."
    if [ -d ".venv" ]; then
        .venv/bin/python scripts/run_etl_nvidia.py 2>&1 | tail -10
    elif [ -d "venv" ]; then
        venv/bin/python scripts/run_etl_nvidia.py 2>&1 | tail -10
    else
        python3 scripts/run_etl_nvidia.py 2>&1 | tail -10
    fi
    success "ETL completed (local)"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Start MLflow Server
# ═════════════════════════════════════════════════════════════════════════════
step "Starting MLflow Tracking Server"

docker compose up -d mlflow 2>&1
wait_for_service "http://localhost:5000" "MLflow" 20
success "MLflow UI → http://localhost:5000"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Train Model
# ═════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_TRAIN" = false ]; then
    step "Training LSTM Model"

    docker compose run --rm -e PYTHONPATH=/app dev \
        bash -c "cd /app && bash scripts/run_training.sh" 2>&1 | tail -20

    if [ -f "models/best_model.pth" ] || [ -f "data/models/checkpoints/best_model.pt" ]; then
        success "Model training completed"
    else
        warn "Model file not found at expected path — checking outputs..."
        find . -name "best_model*" -o -name "*.pth" 2>/dev/null | head -5
    fi
else
    step "Skipping model training (--skip-train)"
    if [ -f "models/best_model.pth" ] || [ -f "data/models/checkpoints/best_model.pt" ]; then
        success "Using existing trained model"
    else
        warn "No trained model found! Consider running without --skip-train"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: Start FastAPI Server
# ═════════════════════════════════════════════════════════════════════════════
step "Starting FastAPI REST API"

docker compose -f docker-compose.api.yml up -d api 2>&1
wait_for_service "http://localhost:8000/health" "FastAPI" 30
success "FastAPI API     → http://localhost:8000"
success "Swagger UI      → http://localhost:8000/docs"

# Quick API smoke test
echo ""
echo "  Running API smoke tests..."
HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null || echo "FAIL")
if echo "$HEALTH" | grep -q "healthy" 2>/dev/null; then
    success "GET /health → healthy"
else
    warn "GET /health → $HEALTH"
fi

DATA=$(curl -sf "http://localhost:8000/data?limit=1" 2>/dev/null || echo "FAIL")
if [ "$DATA" != "FAIL" ] && [ "$DATA" != "" ]; then
    success "GET /data?limit=1 → OK"
else
    warn "GET /data → no response (data may not be loaded yet)"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: Start Streamlit Dashboard
# ═════════════════════════════════════════════════════════════════════════════
step "Starting Streamlit Dashboard"

# Run dashboard as a Docker service
docker run -d \
    --name nvidia-lstm-dashboard \
    --network host \
    -v "$PROJECT_ROOT/data:/app/data:ro" \
    -v "$PROJECT_ROOT/outputs:/app/outputs:ro" \
    -v "$PROJECT_ROOT/mlruns:/app/mlruns:ro" \
    -v "$PROJECT_ROOT/models:/app/models:ro" \
    -v "$PROJECT_ROOT/src:/app/src:ro" \
    -v "$PROJECT_ROOT/configs:/app/configs:ro" \
    -e DATABASE_PATH=/app/data/nvidia_stock.db \
    -e PYTHONPATH=/app \
    $(docker compose images --format json 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    for img in (data if isinstance(data, list) else [data]):
        if 'nvidia-lstm' in str(img.get('Repository', img.get('repository', ''))):
            print(img.get('Repository', img.get('repository', '')) + ':' + img.get('Tag', img.get('tag', 'latest')))
            break
    else:
        print('nvidia-mlops-platform-dev:latest')
except:
    print('nvidia-mlops-platform-dev:latest')
" 2>/dev/null || echo "nvidia-mlops-platform-dev:latest") \
    streamlit run src/dashboard/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --browser.gatherUsageStats=false 2>&1 || true

wait_for_service "http://localhost:8501" "Dashboard" 20
success "Dashboard → http://localhost:8501"

# ═════════════════════════════════════════════════════════════════════════════
# STEP 9: Start Monitoring Stack
# ═════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_MONITORING" = false ]; then
    step "Starting Monitoring Stack (Prometheus + Grafana)"

    docker compose -f docker-compose.monitoring.yml up -d 2>&1
    wait_for_service "http://localhost:9090/-/ready" "Prometheus" 20
    wait_for_service "http://localhost:3000/api/health" "Grafana" 20
    success "Prometheus → http://localhost:9090"
    success "Grafana    → http://localhost:3000 (admin/admin)"
else
    step "Skipping monitoring stack (--skip-monitoring)"
    warn "Prometheus and Grafana not started"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 10: Run Tests
# ═════════════════════════════════════════════════════════════════════════════
step "Running Test Suite"

# Prefer local venv for faster test execution
if [ -d ".venv" ]; then
    .venv/bin/python -m pytest tests/ --tb=short -q 2>&1 | tail -10
elif [ -d "venv" ]; then
    venv/bin/python -m pytest tests/ --tb=short -q 2>&1 | tail -10
else
    # Fallback to Docker
    docker compose run --rm -e PYTHONPATH=/app dev \
        python -m pytest tests/ --tb=short -q 2>&1 | tail -10
fi
success "Test suite completed"

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║                                                           ║"
echo "  ║          🎉  All Services Are Running!  🎉               ║"
echo "  ║                                                           ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║                                                           ║"
echo "  ║   Service            URL                                  ║"
echo "  ║   ─────────────────  ──────────────────────────────────   ║"
echo "  ║   FastAPI API        http://localhost:8000                ║"
echo "  ║   Swagger UI         http://localhost:8000/docs           ║"
echo "  ║   Streamlit          http://localhost:8501                ║"
echo "  ║   MLflow UI          http://localhost:5000                ║"
echo -e "  ║   Prometheus         http://localhost:9090                ║"
echo "  ║   Grafana            http://localhost:3000  (admin/admin) ║"
echo "  ║                                                           ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║                                                           ║"
echo "  ║   Quick demo commands:                                    ║"
echo "  ║     curl localhost:8000/health                            ║"
echo "  ║     curl localhost:8000/data?limit=5                      ║"
echo "  ║     curl -X POST localhost:8000/predict \\                ║"
echo "  ║       -H 'Content-Type: application/json' \\              ║"
echo "  ║       -d '{\"horizon\": 7}'                                ║"
echo "  ║                                                           ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║                                                           ║"
echo "  ║   To stop everything:                                     ║"
echo "  ║     docker compose down                                   ║"
echo "  ║     docker compose -f docker-compose.api.yml down         ║"
echo "  ║     docker compose -f docker-compose.monitoring.yml down  ║"
echo "  ║     docker rm -f nvidia-lstm-dashboard 2>/dev/null        ║"
echo "  ║                                                           ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
