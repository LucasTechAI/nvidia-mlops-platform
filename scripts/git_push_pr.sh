#!/usr/bin/env bash
# =============================================================================
# Create branch, commit, push, and open PR
# Usage: bash scripts/git_push_pr.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BRANCH="feature/local-services-startup"
BASE="main"

# 1. Garante que está na branch base e atualizada
echo "🔄 Atualizando $BASE..."
git checkout "$BASE"
git pull origin "$BASE"

# 2. Cria nova branch
echo "🔀 Criando branch: $BRANCH"
git checkout -b "$BRANCH"

# 3. Stage + commit
echo "📦 Staging e commit..."
git add scripts/run_services.sh
git commit -m "feat(scripts): add run_services.sh for local platform startup

- MLflow UI (:5000) with SQLite backend
- FastAPI (:8000) with uvicorn
- Streamlit dashboard (:8501) with NVIDIA theme
- Prometheus (:9090) via Docker
- Grafana (:3000) via Docker
- Graceful shutdown via Ctrl+C or --stop flag
- Logs written to logs/services/"

# 4. Push
echo "🚀 Push para origin..."
git push -u origin "$BRANCH"

# 5. Abre PR
if command -v gh &>/dev/null; then
    echo "📝 Criando Pull Request..."
    gh pr create \
        --base "$BASE" \
        --title "feat(scripts): local services startup script" \
        --body "## Descrição
Adiciona \`scripts/run_services.sh\` para subir todos os serviços da plataforma localmente.

## Serviços
| Serviço | Porta | Tipo |
|---------|-------|------|
| FastAPI | :8000 | Python (uvicorn) |
| Streamlit | :8501 | Python |
| MLflow | :5000 | Python |
| Prometheus | :9090 | Docker |
| Grafana | :3000 | Docker |

## Como usar
\`\`\`bash
bash scripts/run_services.sh          # Inicia tudo
bash scripts/run_services.sh --stop   # Para tudo
\`\`\`

## Extras
- Auto-detecta virtualenv (\`.venv\` ou \`venv\`)
- Carrega \`.env\` automaticamente
- Graceful shutdown com Ctrl+C
- Logs em \`logs/services/\`"
    echo "✅ PR criado com sucesso!"
else
    REPO_URL=$(git remote get-url origin | sed 's/\.git$//' | sed 's|git@github.com:|https://github.com/|')
    echo "⚠️  gh CLI não encontrado. Crie o PR manualmente:"
    echo "   ${REPO_URL}/compare/${BASE}...${BRANCH}?expand=1"
fi
