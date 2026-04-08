#!/bin/bash
# Start MLflow UI server

set -e

# Get project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DB_PATH="${PROJECT_ROOT}/mlruns/mlflow.db"
ARTIFACT_ROOT="${PROJECT_ROOT}/mlruns/artifacts"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

echo "Starting MLflow UI..."
echo "  Backend store: sqlite:///${DB_PATH}"
echo "  Artifact root: ${ARTIFACT_ROOT}"

# Start MLflow UI with SQLite backend and artifact root
mlflow ui \
    --backend-store-uri "sqlite:///${DB_PATH}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000

echo "MLflow UI is running at http://localhost:5000"
