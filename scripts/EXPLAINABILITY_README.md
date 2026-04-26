# 📊 Explainability Analysis Scripts

This directory contains scripts for generating and analyzing model explainability metrics.

## Available Scripts

### 1. `generate_test_data.py`
Generates test data splits from the trained model's dataset.

```bash
python scripts/generate_test_data.py
```

**Output:**
- `data/processed/X_test.npy` - Test input sequences
- `data/processed/y_test.npy` - Test target values

### 2. `compute_permutation_importance.py`
Computes permutation importance for global feature analysis.

```bash
python scripts/compute_permutation_importance.py
```

**Output:**
- `outputs/explainability/permutation_importance.json` - Raw results
- `outputs/explainability/permutation_importance.png` - Visualization

**What it does:**
- Measures how model performance degrades when each feature is shuffled
- Shows which features contribute most to predictions globally

### 3. `compute_lime.py`
Computes LIME (Local Interpretable Model-agnostic Explanations) for local feature analysis.

```bash
python scripts/compute_lime.py
```

**Output:**
- `outputs/explainability/lime_batch_explanation.json` - Results for 50 samples
- `outputs/explainability/lime_global_importance.png` - Global aggregation chart

**What it does:**
- Explains individual predictions by approximating the model locally
- Provides per-sample feature weights that can be aggregated globally

### 4. `run_explainability.sh` ⭐
**All-in-one script** that runs the complete explainability analysis pipeline.

```bash
./scripts/run_explainability.sh
```

**What it does:**
1. Generates test data
2. Computes permutation importance
3. Computes LIME explanations
4. Displays summary of all results

**Perfect for:**
- First-time setup
- Regenerating all metrics after model retraining
- Quick analysis of a new model

## Quick Start

```bash
# Run everything at once
./scripts/run_explainability.sh

# Or run individually
python scripts/generate_test_data.py
python scripts/compute_permutation_importance.py
python scripts/compute_lime.py
```

## Output Files

All results are saved to `outputs/explainability/`:

| File | Description |
|------|-------------|
| `permutation_importance.json` | Permutation importance raw data |
| `permutation_importance.png` | Permutation importance chart |
| `lime_batch_explanation.json` | LIME results with per-sample weights |
| `lime_global_importance.png` | LIME global feature importance chart |
| `EVALUATION_REPORT.md` | Comprehensive analysis report |

## Understanding the Results

### Permutation Importance
- **Higher values = more important features**
- Shows global impact: how much performance drops when a feature is removed
- Good for understanding overall feature contributions

### LIME
- **Higher mean absolute weights = more important features**
- Shows local impact: how features influence individual predictions
- Good for understanding model behavior on specific examples

### Both Methods
- Use both for comprehensive understanding
- They should generally agree on feature rankings
- Differences reveal global vs local importance patterns

## View Results

```bash
# View comprehensive report
cat outputs/explainability/EVALUATION_REPORT.md

# View raw JSON data
cat outputs/explainability/permutation_importance.json
cat outputs/explainability/lime_batch_explanation.json

# Open visualizations (if using GUI)
xdg-open outputs/explainability/permutation_importance.png
xdg-open outputs/explainability/lime_global_importance.png
```

## Dashboard Integration

These scripts generate the data needed for the dashboard's Explainability tab:
1. Run `./scripts/run_explainability.sh`
2. Open the Next.js dashboard at http://localhost:3001
3. Navigate to **Evaluation** (`/evaluation`) → **Explainability** tab
4. Click "🔬 Compute Feature Importance" to see permutation results
5. Saved LIME plots will be displayed automatically

## Troubleshooting

**Error: "Test data not found"**
```bash
# Generate test data first
python scripts/generate_test_data.py
```

**Error: "Model not found"**
```bash
# Train a model first
./scripts/run_training.sh
```

**Error: "Module not found"**
```bash
# Activate virtual environment
source .venv/bin/activate
```
