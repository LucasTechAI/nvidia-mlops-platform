# Model Card — NVIDIA Stock Prediction LSTM

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | NVIDIA Stock Price LSTM |
| **Version** | 1.0.0 |
| **Type** | Time Series Forecasting |
| **Architecture** | LSTM (Long Short-Term Memory) |
| **Framework** | PyTorch 2.6+ |
| **Developed by** | Lucas (Datathon Fase 05) |
| **Date** | 2025 |
| **License** | MIT |

## Model Architecture

```
Input (5 features × 60 timesteps)
  → LSTM Layer 1 (hidden_size=128, dropout=0.2)
  → LSTM Layer 2 (hidden_size=128, dropout=0.2)
  → Fully Connected (128 → 5)
  → Output (5 features: Open, High, Low, Close, Volume)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| input_size | 5 |
| hidden_size | 128 |
| num_layers | 2 |
| dropout | 0.2 |
| sequence_length | 60 |
| batch_size | 32 |
| learning_rate | 0.001 |
| optimizer | Adam |
| loss_function | MSE |
| early_stopping_patience | 10 |
| max_epochs | 100 |

## Intended Use

### Primary Use Case
- Short-term stock price forecasting for NVIDIA (NVDA) ticker
- Generating forecasts with uncertainty estimates (Monte Carlo Dropout)
- Educational and analytical purposes in financial ML

### Out-of-Scope Uses
- ❌ Making real investment decisions without professional advice
- ❌ High-frequency trading
- ❌ Predicting other stocks without retraining
- ❌ Long-term forecasting (>30 days)

## Training Data

| Field | Value |
|-------|-------|
| **Source** | Yahoo Finance (yfinance API) |
| **Ticker** | NVDA |
| **Period** | 2017–present |
| **Features** | Open, High, Low, Close, Volume |
| **Preprocessing** | MinMaxScaler (0–1 normalization) |
| **Split** | 70% train / 15% validation / 15% test |

### Data Limitations
- Data from a single source (Yahoo Finance)
- Does not account for after-hours trading
- Stock splits may cause distribution shifts
- Market regime changes not explicitly handled

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |
| **R²** | Coefficient of Determination |

Metrics are computed on the held-out test set (15% of data) and tracked via MLflow.

## Uncertainty Estimation

Monte Carlo Dropout is used at inference time:
- Dropout remains active during prediction
- N samples (default=100) are generated
- Mean prediction and confidence intervals are computed
- Configurable confidence level (default=95%)

## Ethical Considerations

### Risks
- **Financial risk**: Users may interpret predictions as investment advice
- **Overfitting**: Model may overfit to historical patterns that don't repeat
- **Black swan events**: Cannot predict unprecedented market events
- **Bias**: Training data reflects historical market conditions

### Mitigations
- Risk disclaimers added to all prediction outputs
- Uncertainty estimates accompany all predictions
- Documentation clearly states limitations
- Guardrails prevent the system from giving investment recommendations

## Monitoring & Drift Detection

- **PSI (Population Stability Index)** monitors data distribution shifts
  - PSI > 0.1 → Warning alert
  - PSI > 0.2 → Retrain trigger
- **Prometheus metrics** track latency, error rates, and throughput
- **MLflow** logs all training runs for reproducibility

## Citation

```bibtex
@misc{nvidia-lstm-forecast,
  title={NVIDIA Stock Price Prediction with LSTM},
  author={Lucas},
  year={2025},
  note={Datathon Fase 05 — FIAP Pós Tech MLOps}
}
```
