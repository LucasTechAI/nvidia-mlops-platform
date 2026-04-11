#!/usr/bin/env bash
# =============================================================================
# NVIDIA MLOps Platform — Start All Services (Local, sem Docker)
# =============================================================================
# Sobe todos os serviços localmente usando o virtualenv do projeto:
#   FastAPI  :8000  |  Streamlit :8501  |  MLflow :5000
#   Prometheus :9090  |  Grafana :3000
#
# Usage:
#   bash scripts/run_services.sh            # Inicia tudo
#   bash scripts/run_services.sh --stop     # Para tudo
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

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PIDFILE="$PROJECT_ROOT/.services.pid"
LOGDIR="$PROJECT_ROOT/logs/services"
mkdir -p "$LOGDIR"

# ─── Detect Python ───────────────────────────────────────────────────────────
if [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
    PIP="$PROJECT_ROOT/.venv/bin/pip"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
    PIP="$PROJECT_ROOT/venv/bin/pip"
else
    PYTHON="python3"
    PIP="pip3"
fi

# ─── Load .env ───────────────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# MLflow backend — usa SQLite (não filesystem depreciado)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$PROJECT_ROOT/mlruns/mlflow.db}"

# ─── Stop function ───────────────────────────────────────────────────────────
stop_services() {
    echo ""
    echo -e "${BOLD}${YELLOW}  Stopping all services...${NC}"

    if [ -f "$PIDFILE" ]; then
        while IFS='=' read -r name pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Stopped $name (PID $pid)"
            fi
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi

    # Kill any leftover processes on our ports
    for port in 8000 3001 5000; do
        pid=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Stop Prometheus & Grafana containers if running (both old and new names)
    docker rm -f nvidia-prometheus nvidia-grafana prometheus grafana 2>/dev/null || true

    echo -e "  ${GREEN}✅ All services stopped${NC}"
    echo ""
    exit 0
}

# Handle --stop flag
if [[ "${1:-}" == "--stop" ]]; then
    stop_services
fi

# ─── Cleanup on exit ────────────────────────────────────────────────────────
trap stop_services INT TERM

# ─── Kill any process already using our ports ────────────────────────────────
MYPID=$$
for port in 5000 8000 3001 9090 3000; do
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "  ${YELLOW}⚠ Port $port in use — killing...${NC}"
        for p in $pids; do
            [ "$p" != "$MYPID" ] && kill -9 "$p" 2>/dev/null || true
        done
        sleep 0.5
    fi
done

# ─── Stop any existing services first ────────────────────────────────────────
if [ -f "$PIDFILE" ]; then
    echo -e "${YELLOW}  Stopping previous services...${NC}"
    stop_services 2>/dev/null || true
fi

# ─── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║                                                           ║"
echo "  ║     🟢  NVIDIA MLOps Platform — Local Services  🟢        ║"
echo "  ║                                                           ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── Helper ──────────────────────────────────────────────────────────────────
wait_for() {
    local url=$1 name=$2 max=${3:-20} i=0
    printf "  Waiting for %-12s " "$name"
    while [ $i -lt $max ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            return 0
        fi
        printf "."
        sleep 1
        i=$((i + 1))
    done
    echo -e "${YELLOW} (timeout — may still be starting)${NC}"
    return 0
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. MLflow UI  (:5000)
# ═════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}  [1/5] Starting MLflow UI...${NC}"
$PYTHON -m mlflow ui \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --host 0.0.0.0 \
    --port 5000 \
    > "$LOGDIR/mlflow.log" 2>&1 &
echo "mlflow=$!" >> "$PIDFILE"
wait_for "http://localhost:5000" "MLflow"

# ═════════════════════════════════════════════════════════════════════════════
# 2. FastAPI  (:8000)
# ═════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}  [2/5] Starting FastAPI API...${NC}"
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT" $PYTHON -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > "$LOGDIR/fastapi.log" 2>&1 &
echo "fastapi=$!" >> "$PIDFILE"
wait_for "http://localhost:8000/health" "FastAPI"

# ═════════════════════════════════════════════════════════════════════════════
# 3. Next.js Dashboard  (:3001)
# ═════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}  [3/5] Starting Next.js Dashboard...${NC}"
cd "$PROJECT_ROOT/dashboard-frontend"
if [ ! -d "node_modules" ]; then
    echo -e "  ${YELLOW}Installing frontend dependencies...${NC}"
    npm install > "$LOGDIR/npm-install.log" 2>&1
fi
# Build if .next doesn't exist
if [ ! -d ".next" ]; then
    echo -e "  ${YELLOW}Building frontend (first time)...${NC}"
    npx next build > "$LOGDIR/nextjs-build.log" 2>&1
fi
npm run start > "$LOGDIR/nextjs.log" 2>&1 &
echo "nextjs=$!" >> "$PIDFILE"
cd "$PROJECT_ROOT"
wait_for "http://localhost:3001" "Next.js" 30

# ═════════════════════════════════════════════════════════════════════════════
# 4. Prometheus  (:9090)  — via Docker
# ═════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}  [4/5] Starting Prometheus...${NC}"
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    docker rm -f nvidia-prometheus prometheus 2>/dev/null || true
    # Kill anything on port 9090
    for p in $(lsof -ti :9090 2>/dev/null); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null || true; done
    sleep 0.5
    if docker run -d \
        --name nvidia-prometheus \
        -p 9090:9090 \
        -v "$PROJECT_ROOT/configs/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
        --add-host=host.docker.internal:host-gateway \
        prom/prometheus:latest \
        --config.file=/etc/prometheus/prometheus.yml \
        --web.enable-lifecycle \
        > /dev/null 2>&1; then
        wait_for "http://localhost:9090/-/ready" "Prometheus"
    else
        echo -e "  ${YELLOW}⚠ Prometheus container failed to start — skipping${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ Docker not available — skipping Prometheus${NC}"
fi

# ═════════════════════════════════════════════════════════════════════════════
# 5. Grafana  (:3000)  — via Docker
# ═════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}  [5/5] Starting Grafana...${NC}"
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    docker rm -f nvidia-grafana grafana 2>/dev/null || true
    # Kill anything on port 3000
    for p in $(lsof -ti :3000 2>/dev/null); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null || true; done
    sleep 0.5
    if docker run -d \
        --name nvidia-grafana \
        -p 3000:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD=admin \
        -v "$PROJECT_ROOT/configs/grafana/provisioning:/etc/grafana/provisioning:ro" \
        -v "$PROJECT_ROOT/configs/grafana/dashboards:/var/lib/grafana/dashboards:ro" \
        --add-host=host.docker.internal:host-gateway \
        grafana/grafana:latest \
        > /dev/null 2>&1; then
        wait_for "http://localhost:3000/api/health" "Grafana"
    else
        echo -e "  ${YELLOW}⚠ Grafana container failed to start — skipping${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ Docker not available — skipping Grafana${NC}"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${GREEN}  ✅ All services running!${NC}"
echo ""
echo -e "  ${CYAN}FastAPI${NC}      http://localhost:8000"
echo -e "  ${CYAN}Swagger${NC}      http://localhost:8000/docs"
echo -e "  ${CYAN}Next.js${NC}      http://localhost:3001"
echo -e "  ${CYAN}MLflow${NC}       http://localhost:5000"
echo -e "  ${CYAN}Prometheus${NC}   http://localhost:9090"
echo -e "  ${CYAN}Grafana${NC}      http://localhost:3000  (admin/admin)"
echo ""
echo -e "  Logs: ${YELLOW}logs/services/*.log${NC}"
echo -e "  Stop: ${YELLOW}bash scripts/run_services.sh --stop${NC} or Ctrl+C"
echo ""

# Keep script running (so Ctrl+C triggers the trap)
echo -e "  ${CYAN}Press Ctrl+C to stop all services${NC}"
echo ""
wait
