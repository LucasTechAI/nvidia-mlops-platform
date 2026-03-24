.PHONY: install test lint format train serve dashboard data docker-build docker-up clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies
	pip install -r requirements.txt
	pip install ruff mypy bandit pytest-cov pandera pre-commit

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=60

lint: ## Run linter (ruff)
	ruff check src/ tests/

format: ## Format code (ruff)
	ruff format src/ tests/

typecheck: ## Run type checker (mypy)
	mypy src/ --ignore-missing-imports

security: ## Run security scan (bandit)
	bandit -r src/ -ll

train: ## Run model training
	python scripts/run_training.sh

serve: ## Start FastAPI API server
	bash scripts/run_api.sh

dashboard: ## Start Streamlit dashboard
	bash scripts/run_dashboard.sh

data: ## Run ETL pipeline
	python scripts/run_etl_nvidia.py

hpo: ## Run hyperparameter optimization
	bash scripts/run_hpo.sh

predict: ## Run prediction
	bash scripts/run_prediction.sh

mlflow-ui: ## Start MLflow UI
	bash scripts/start_mlflow_ui.sh

docker-build: ## Build Docker images
	docker compose build

docker-up: ## Start all services via Docker Compose
	docker compose up -d

docker-down: ## Stop all Docker services
	docker compose down

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

all: lint typecheck security test ## Run all quality checks
