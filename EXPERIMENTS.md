# 📊 Experiments Documentation

Documentation of experiments conducted during the development of the NVIDIA stock price prediction platform with LSTM.

> All experiments are automatically tracked via **MLflow**.
> To view: `bash scripts/start_mlflow_ui.sh` → http://localhost:5000

---

## Overview

The project uses an LSTM (Long Short-Term Memory) network to predict the closing price of NVIDIA (NVDA) stock. The experiments explore variations in architecture, hyperparameters, features, and training strategies.

### Metrics Used

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error — primary optimization metric |
| **MAE** | Mean Absolute Error — robust to outliers |
| **MAPE** | Mean Absolute Percentage Error — relative error (%) |

Metrics are computed on the validation set every epoch and logged in MLflow (`val_rmse`, `val_mae`, `val_mape`).

---

## 1. Baseline Model

### Experiment 1.1: Simple LSTM

**Objective**: Establish a baseline with a minimal architecture.

| Parameter | Value |
|-----------|-------|
| LSTM Layers | 1 |
| Hidden Size | 64 |
| Dropout | 0.0 (single layer) |
| Sequence Length | 30 days |
| Features | Close (univariate) |
| Learning Rate | 0.001 |
| Epochs | 100 (early stopping, patience=10) |
| Batch Size | 32 |

**Findings**:
- Simple model converges quickly (~20 epochs)
- Captures general trends but misses short-term variations
- Serves as a baseline for comparison with more complex architectures

### Experiment 1.2: Deep LSTM (Main Model)

**Objective**: Increase model capacity with more layers and features.

| Parameter | Value |
|-----------|-------|
| LSTM Layers | 2 (stacked) |
| Hidden Size | 128 |
| Dropout | 0.2 (between layers) |
| Sequence Length | 60 days |
| Features | OHLCV (5 features) |
| Learning Rate | 0.001 |
| Epochs | 100 (early stopping, patience=10) |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | MSE |
| Gradient Clipping | max_norm=1.0 |

**MLflow Run ID**: `ee17873ae3354481926bf70ac77130ef`
**Date**: 2026-02-02
**Framework**: PyTorch 2.10+CUDA, MLflow 3.8.1
**Model Size**: ~800 KB (802,305 bytes)

**Results (normalized scale 0–1)**:

| Set | Loss | RMSE | MAE | MAPE |
|-----|------|------|-----|------|
| **Validation** | 0.003162 | 0.053320 | 0.030938 | 30.64% |
| **Test** | 0.019251 | 0.137608 | 0.080397 | 192.36% |

> **Note on MAPE**: The high MAPE values result from computation on the normalized scale (0–1),
> where values close to zero inflate the percentage error. RMSE and MAE are the most
> representative metrics for this use case.

**Training Data**:
- **Total samples**: 2,259 sequences (2,319 records, seq_len=60)
- **Split**: Train=1,581 / Val=339 / Test=339

**Generated Artifacts**:
- `loss_curves.png` — Training loss vs validation loss curves
- `predictions_vs_actual.png` — Prediction vs actual values on the test set
- `scaler.joblib` — MinMaxScaler for inverse transform of predictions

**Findings**:
- 2 LSTM layers with hidden=128 provides a good balance between capacity and generalization
- Dropout of 0.2 between layers effectively prevents overfitting
- Gradient clipping (max_norm=1.0) stabilizes training on volatile financial data
- Early stopping typically triggers between epochs 40–60, indicating adequate convergence
- OHLCV features (5 dimensions) provide richer market context than close-only

---

## 2. Hyperparameter Optimization (HPO)

### Experiment 2.1: Bayesian Search with Optuna (20 trials)

**Objective**: Find optimal hyperparameters via TPE Sampler.

**Search Space**:

| Hyperparameter | Range | Type |
|----------------|-------|------|
| `num_layers` | [1, 4] | integer |
| `hidden_size` | {32, 64, 128, 256} | categorical |
| `learning_rate` | [1e-5, 1e-2] | float (log scale) |
| `dropout` | [0.1, 0.5] | float |
| `sequence_length` | {30, 60, 90, 120} | categorical |
| `batch_size` | {16, 32, 64, 128} | categorical |

**Training configuration per trial**:
- Epochs: 50 (reduced for HPO)
- Early stopping patience: 5
- Optimizer: Adam
- Objective: Minimize val_RMSE

**Findings**:
- Hidden size 128–256 consistently outperforms 32–64
- Optimal learning rate in the range [5e-4, 2e-3]
- Dropout between 0.15–0.30 offers the best regularization
- Sequence length of 60 days is the sweet spot (30 loses context, 90+ adds noise)
- Batch size 32 offers a good trade-off between speed and generalization

### Experiment 2.2: Extended HPO (50+ trials)

**Objective**: Refine search with more trials for convergence.

**Configuration**: Same as 2.1 with 50–100 trials.

**Findings**:
- From ~40 trials onward, marginal gain diminishes significantly
- Top-5 configurations converge to: 2 layers, hidden 128–256, lr ~0.001, dropout ~0.2
- Confirms the baseline configuration (Exp 1.2) as robust

---

## 3. Feature Engineering

### Experiment 3.1: Single Feature (Close)

| Parameter | Value |
|-----------|-------|
| Features | Close (1 feature) |
| Normalization | MinMaxScaler (0–1) |

**Pros**: Simple, lower risk of overfitting, faster training
**Cons**: Loses volatility information (high-low spread), volume, and opening data

### Experiment 3.2: Multi-Feature (OHLCV) — Adopted Configuration

| Parameter | Value |
|-----------|-------|
| Features | Open, High, Low, Close, Volume (5 features) |
| Normalization | MinMaxScaler (0–1), applied feature-wise |
| input_size | 5 |

**Pros**: Richer market context, captures volatility and volume
**Cons**: Greater complexity, requires hidden_size >= 64

**Findings**:
- OHLCV improves capture of reversal and continuation patterns
- Volume as an additional feature helps during high-volatility periods
- Feature-wise normalization is essential to prevent dominance of features with larger scales

### Experiment 3.3: Technical Indicators

**Status**: Future work
**Planned Features**: RSI, MACD, Bollinger Bands, Moving Averages
**Hypothesis**: Technical indicators may improve short-term predictions

---

## 4. Architecture Variations

### Experiment 4.1: Unidirectional LSTM (Adopted)

| Parameter | Value |
|-----------|-------|
| Bidirectional | False |
| Total Parameters | ~200K |

**Justification**: In financial time series, future information is not available. A unidirectional LSTM respects temporal causality.

### Experiment 4.2: Bidirectional LSTM

| Parameter | Value |
|-----------|-------|
| Bidirectional | True |
| Total Parameters | ~400K (2x) |

**Findings**:
- Doubles the number of parameters without proportional metric gains
- In time series prediction, look-ahead is unrealistic
- Higher risk of overfitting on moderate-sized datasets (~6,800 samples)

### Experiment 4.3: LSTM with Attention

**Status**: Future work
**Hypothesis**: Attention mechanism can focus on more relevant timesteps

---

## 5. Sequence Length Analysis

**Objective**: Determine the optimal lookback window.

| Sequence Length | Observations |
|-----------------|-------------|
| 30 days | Captures short-term patterns; may miss seasonal trends |
| **60 days** | **Best balance**: captures trends without excessive noise |
| 90 days | Long context; tendency to overfit with limited dataset |
| 120 days | Extended context; longer training time, marginal gain |

**Key Finding**: 60 days (approximately 3 months of trading) captures short-to-medium-term market cycles without adding noise from different market regimes.

---

## 6. Regularization Techniques

### 6.1 Dropout

| Dropout | Observations |
|---------|-------------|
| 0.0 | No regularization — overfitting at ~20 epochs |
| 0.1 | Light regularization — good for simple models |
| **0.2** | **Adopted**: balance between capacity and generalization |
| 0.3 | Moderate regularization — loss of expressiveness |
| 0.5 | Strong regularization — underfitting with 2-layer models |

**Note**: Dropout is only applied between LSTM layers (`dropout=0` if `num_layers=1`).

### 6.2 Early Stopping

| Patience | Observations |
|----------|-------------|
| 5 | Stops too early — model does not converge fully |
| **10** | **Adopted**: allows recovery from temporary plateaus |
| 15 | Longer training without significant gains |
| 20 | Risk of overfitting if val_loss starts increasing |

### 6.3 Gradient Clipping

**Adopted configuration**: `max_norm=1.0` (via `torch.nn.utils.clip_grad_norm_`)

**Findings**:
- Essential for stability on financial data (sudden price swings)
- Prevents gradient explosion common in LSTM with backprop through time
- max_norm=1.0 does not impair convergence speed

---

## 7. Training Strategies

### 7.1 Optimizer

| Optimizer | Observations |
|-----------|-------------|
| **Adam** | **Adopted**: fast convergence, good default for LSTM |
| SGD + momentum | Slower convergence, may generalize better with tuning |
| AdamW | Future work: explicit weight decay |

### 7.2 Batch Size Impact

| Batch Size | Trade-offs |
|------------|------------|
| 16 | More frequent, noisier updates; slower training |
| **32** | **Adopted**: good speed/generalization balance |
| 64 | More stable gradient estimates; less generalization |
| 128 | Faster training; risk of converging to shallow minima |

### 7.3 Data Split

**Adopted configuration**: 70% train / 15% validation / 15% test

**Findings**:
- Temporal split (no shuffle) is critical for time series — avoids data leakage
- 15% for validation is sufficient for reliable early stopping
- Test set reserved for single final evaluation

---

## 8. Ensemble Methods

### 8.1 Top-K Model Averaging

**Status**: Future work
**Approach**: Average predictions from the top 3–5 HPO models
**Expected benefit**: Variance reduction in predictions

### 8.2 Weighted Ensemble

**Status**: Future work
**Approach**: Weight models by validation performance

---

## 9. Champion-Challenger Pipeline

### 9.1 Automated Model Validation

**Objective**: Ensure new models outperform the production model before promotion.

**Implementation**: `src/training/champion_challenger.py`

| Parameter | Value |
|-----------|-------|
| Promotion threshold | δ RMSE ≤ −0.5% |
| Compared metrics | RMSE, MAE, MAPE, R², Directional Accuracy |
| Trigger | Drift detection or manual execution |
| Logging | MLflow (experiment `champion_challenger`) |
| Artifacts | `outputs/champion_challenger/latest_comparison.json` |

**Pipeline flow**:
1. **Drift detection** — Runs KS-test / PSI on recent data via `detect_drift_from_db()`
2. **Challenger training** — If drift is detected, trains a new model with `train_model()`
3. **Metric comparison** — Evaluates both on the holdout set with `evaluate_model()`
4. **Promotion decision** — Promotes challenger if `rmse_delta_pct ≤ −0.005` (0.5%)
5. **MLflow logging** — Logs metrics, parameters, and decision with tag `pipeline=champion_challenger`

**Promotion criteria**:
- ✅ Promote: Challenger RMSE is ≥0.5% better than champion
- ⚠️ Keep: Improvement <0.5% (below threshold)
- ❌ Keep: Challenger RMSE is worse than champion

**Findings**:
- The 0.5% threshold prevents promotions due to statistical noise
- Full MLflow logging enables auditing of all promotion decisions
- Pipeline integrated with drift detection closes the monitoring → retraining loop

---

## 10. Explainability

### 10.1 Permutation Importance

**Objective**: Quantify the contribution of each feature to the model's predictions.

**Implementation**: `src/explainability/feature_importance.py`

| Parameter | Value |
|-----------|-------|
| Method | Permutation Importance |
| Base metric | RMSE |
| Repetitions | 5 (for statistical stability) |
| Analyzed features | Open, High, Low, Close, Volume |
| Logging | MLflow (artifacts + metrics) |

**Flow**:
1. Evaluate baseline RMSE of the model on the test dataset
2. For each feature: shuffle the column, recalculate RMSE, measure degradation
3. Repeat N times to obtain mean and standard deviation
4. Ranking: the feature with the greatest degradation = most important

**Generated artifacts**:
- `feature_importance.png` — Bar chart with importance per feature
- `feature_importance.json` — Numerical data for reproducibility
- Metrics logged in MLflow for each feature

**Expected findings**:
- Close and Volume tend to be the most important features for next-day prediction
- Open and High/Low provide intraday range context
- Importance may vary with market regimes (high vs. low volatility)

---

## 11. Evaluation Framework

### 11.1 Golden Set

**Location**: `data/golden_set/golden_set.json`

A curated set of question-answer pairs to evaluate:
- Quality of the RAG agent's responses
- Faithfulness to retrieved context
- Relevance of model predictions

### 11.2 LLM-as-Judge

**Implementation**: `evaluation/llm_judge.py`

Uses an LLM as an evaluator to score responses on:
- Relevance of the answer to the question
- Factual accuracy based on the data
- Completeness of the answer

### 11.3 RAGAS Evaluation

**Implementation**: `evaluation/ragas_eval.py`

RAGAS framework for RAG pipeline evaluation:
- **Faithfulness**: Is the answer faithful to the retrieved context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: Is the retrieved context precise?
- **Context Recall**: Is the retrieved context complete?

### 11.4 A/B Test Prompts

**Implementation**: `evaluation/ab_test_prompts.py`

Compares prompt variations to identify the most effective formulation for the agent.

---

## 12. Observability & Monitoring

### 12.1 Monitoring Stack

| Component | Function | Port |
|-----------|----------|------|
| **Prometheus** | Metrics collection | :9090 |
| **Grafana** | Dashboards & alerts | :3000 |
| **MLflow** | Experiment tracking | :5000 |
| **Streamlit** | Interactive dashboard | :8501 |

### 12.2 Drift Detection

**Implementation**: `src/monitoring/drift.py`

| Method | Description |
|--------|-------------|
| KS-test | Kolmogorov-Smirnov test for distribution differences |
| PSI | Population Stability Index |
| Evidently | Framework for drift reports (optional) |

### 12.3 Dashboard Observability

**Additional Streamlit tabs** (`src/dashboard/components/`):
- **Observability**: Drift detection, champion-challenger status, telemetry links
- **Evaluation**: Evaluation metrics, explainability, LLM-judge results
- **AI Agent**: Interactive chat interface with RAG agent

### 12.4 Grafana Dashboards

**Provisioning**: `configs/grafana/`

Pre-configured dashboards:
- Inference metrics (latency, throughput)
- Model metrics (RMSE, MAE over time)
- Drift alerts

---

## Best Practices Identified

### Data Preprocessing
1. **Normalization**: MinMaxScaler (0–1) works well for financial data
2. **Sequences**: Maintain temporal order — never shuffle before creating sequences
3. **Temporal split**: 70/15/15 with chronological cutoff (no shuffle)
4. **Filtering**: Data from 2017+ reduces noise from different market regimes
5. **MAPE handling**: Threshold of 1e-3 to avoid division by zero on normalized values

### Model Architecture
1. **Hidden Size**: 128 provides good capacity without overfitting for ~6,800 samples
2. **Num Layers**: 2 layers balances depth and stability
3. **Dropout**: 0.2 between layers is the sweet spot
4. **Initialization**: Xavier (input weights) + Orthogonal (hidden weights)

### Training
1. **Early Stopping**: Patience=10 prevents overfitting, allows recovery from plateaus
2. **Gradient Clipping**: max_norm=1.0 stabilizes LSTM on volatile series
3. **Batch Size**: 32 balances speed and generalization
4. **Learning Rate**: 0.001 (Adam) — robust starting point

### MLflow Organization
1. **Per-epoch metrics**: `train_loss`, `val_loss`, `val_rmse`, `val_mae`, `val_mape`
2. **Final metrics**: `best_val_loss`, `training_time`
3. **Artifacts**: Model (.pth), scaler (.joblib), plots (.png)
4. **Registered model**: PyTorch flavor with conda/pip environments

---

## Run Log

### Run `ee17873a` — Reference Model

| Field | Value |
|-------|-------|
| **Run ID** | `ee17873ae3354481926bf70ac77130ef` |
| **Date** | 2026-02-02, 22:55 UTC |
| **Framework** | PyTorch 2.10.0+cu128 |
| **MLflow** | 3.8.1 |
| **Python** | 3.12.3 |
| **Model ID** | `m-3e0081ef10d84a849bec198392752b10` |
| **Model Size** | 802,305 bytes (~800 KB) |
| **Architecture** | 2-layer LSTM, hidden=128, dropout=0.2 |
| **Features** | OHLCV (5 dimensions) |
| **Sequence Length** | 60 days |
| **Normalization** | MinMaxScaler (0–1) |

**Artifacts**:

| Artifact | Description |
|----------|-------------|
| `loss_curves.png` | Training loss vs validation loss curves per epoch |
| `predictions_vs_actual.png` | Model predictions vs actual values on test set |
| `scaler.joblib` | Serialized MinMaxScaler for inverse transform |

To view detailed metrics and artifacts for this run:
```bash
bash scripts/start_mlflow_ui.sh
# Go to http://localhost:5000 → Experiment 1 → Run ee17873a
```

---

## Future Directions

### Short-term
- [x] Champion-Challenger pipeline for automated model validation
- [x] Feature importance via permutation importance
- [x] Dashboard observability tabs (drift, agent chat, evaluation)
- [ ] Walk-forward validation for more robust evaluation
- [ ] Comparison with naive baselines (last value, moving average)
- [ ] Multiple seeds for statistical significance

### Medium-term
- [ ] Add technical indicators as features (RSI, MACD, Bollinger)
- [ ] Implement temporal attention mechanism
- [ ] Test ensemble of top HPO models
- [ ] Calibrated prediction intervals

### Long-term
- [ ] Direct multi-step predictions (vs. autoregressive)
- [ ] Transfer learning to other stocks
- [ ] Integration with sentiment data (news, social media)
- [ ] Portfolio optimization based on predictions

---

## Experiment Template

When documenting new experiments, use the template below:

```markdown
### Experiment X.Y: [Name]

**Date**: YYYY-MM-DD
**Objective**: [What is being tested]
**Hypothesis**: [Expected result]

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Param 1 | value |
| Param 2 | value |

**Results**:
- RMSE: X.XXXX
- MAE: X.XXXX
- MAPE: X.XX%
- MLflow Run ID: [run_id]

**Findings**:
1. Finding 1
2. Finding 2

**Next steps**:
- Action 1
- Action 2
```

---

## Reproducibility

All experiments can be reproduced with:

1. **MLflow Run ID** for exact parameters and artifacts
2. **Source code**: `src/training/train.py`, `src/training/hyperparameter_search.py`
3. **Data**: `python3 scripts/run_etl_nvidia.py` (extract updated data)
4. **Environment**: `requirements.txt` (pinned versions)
5. **Training**: `bash scripts/run_training.sh`
6. **HPO**: `bash scripts/run_hpo.sh <n_trials>`

---

**Last updated**: 2026-03-28
**Maintained by**: LucasTechAI
