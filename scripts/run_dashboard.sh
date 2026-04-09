#!/bin/bash
# =============================================================================
# Run Next.js Dashboard
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/dashboard-frontend"

# Default port
PORT=${1:-3001}

echo -e "${GREEN}Starting NVIDIA MLOps Dashboard (Next.js)...${NC}"
echo -e "${YELLOW}Dashboard will be available at: http://localhost:${PORT}${NC}"
echo ""

# Check if node_modules exist
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${CYAN}Installing dependencies...${NC}"
    cd "$FRONTEND_DIR" && npm install
fi

# Start Next.js dev server
cd "$FRONTEND_DIR"

if [ "${NODE_ENV:-}" = "production" ]; then
    echo -e "${CYAN}Building for production...${NC}"
    npm run build
    PORT=$PORT npm run start
else
    PORT=$PORT npm run dev
fi
