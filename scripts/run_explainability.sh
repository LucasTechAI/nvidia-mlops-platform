#!/bin/bash
# Run complete explainability analysis
# This script generates test data and computes all explainability metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "🔬 Running Complete Explainability Analysis"
echo "=========================================="
echo ""

# Step 1: Generate test data
echo "📦 Step 1/3: Generating test data..."
$VENV_PYTHON "$SCRIPT_DIR/generate_test_data.py"
echo ""

# Step 2: Compute permutation importance
echo "🔀 Step 2/3: Computing permutation importance..."
$VENV_PYTHON "$SCRIPT_DIR/compute_permutation_importance.py" | tail -15
echo ""

# Step 3: Compute LIME explanations
echo "🔍 Step 3/3: Computing LIME explanations..."
$VENV_PYTHON "$SCRIPT_DIR/compute_lime.py" | tail -15
echo ""

echo "=========================================="
echo "✅ Explainability analysis complete!"
echo ""
echo "📁 Results saved to: $PROJECT_ROOT/outputs/explainability/"
echo ""
echo "Generated files:"
ls -lh "$PROJECT_ROOT/outputs/explainability/" | grep -E '\.(png|json|md)$' | awk '{print "  - " $9 " (" $5 ")"}'
echo ""
echo "📊 View the comprehensive report:"
echo "   cat $PROJECT_ROOT/outputs/explainability/EVALUATION_REPORT.md"
