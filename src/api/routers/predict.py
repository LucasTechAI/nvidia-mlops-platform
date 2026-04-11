"""
Prediction endpoints for generating forecasts.
"""

import time as _time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import ModelState, get_model_state
from src.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)
from src.etl.preprocessing import load_data_from_db
from src.monitoring.metrics import PREDICTION_ERRORS, track_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])


def get_forecast_dates(last_date: pd.Timestamp, horizon: int) -> List[datetime]:
    """Generate trading day dates (skip weekends)."""
    forecast_dates = []
    current_date = pd.Timestamp(last_date)

    while len(forecast_dates) < horizon:
        current_date = current_date + pd.Timedelta(days=1)
        if current_date.weekday() < 5:  # Skip weekends
            forecast_dates.append(current_date.to_pydatetime())

    return forecast_dates


def generate_forecast_with_uncertainty(
    model: torch.nn.Module,
    initial_sequence: torch.Tensor,
    horizon: int,
    n_samples: int,
    device: str,
) -> tuple:
    """Generate forecast with Monte Carlo Dropout for uncertainty.

    Returns Close price predictions (index 3) from multi-feature OHLCV output.
    """
    model.train()  # Enable dropout
    close_idx = 3  # Close column index in OHLCV

    all_predictions = []

    for _ in range(n_samples):
        sequence = initial_sequence.clone()
        predictions = []

        for _ in range(horizon):
            with torch.no_grad():
                pred = model(sequence)
            pred_np = pred.cpu().numpy().flatten()
            # Extract Close price (index 3) for the prediction series
            close_val = pred_np[close_idx] if len(pred_np) > close_idx else pred_np[0]
            predictions.append(close_val)

            # Sliding window update (use full multi-feature output)
            new_input = pred.unsqueeze(0)
            sequence = torch.cat([sequence[:, 1:, :], new_input], dim=1)

        all_predictions.append(predictions)

    model.eval()
    all_predictions = np.array(all_predictions)

    mean_preds = np.mean(all_predictions, axis=0)
    std_preds = np.std(all_predictions, axis=0)

    return mean_preds, std_preds


@router.post("", response_model=PredictResponse)
@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest, state: ModelState = Depends(get_model_state)) -> PredictResponse:
    """
    Generate stock price predictions for the next N days.

    Uses the best trained LSTM model with optional Monte Carlo Dropout
    for uncertainty estimation.
    """
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call /health to check status.",
        )

    _start = _time.time()
    try:
        # Load historical data
        df = load_data_from_db(start_year=2017)

        # Capitalize columns to match training convention (ETL returns lowercase)
        rename_map = {col: col.capitalize() for col in df.columns if col.islower()}
        if rename_map:
            df = df.rename(columns=rename_map)

        df["date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Prepare multi-feature sequence (OHLCV — same as training)
        sequence_length = state.model_config.get("sequence_length", 60)
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_features = [col for col in feature_columns if col in df.columns]
        feature_data = df[available_features].values
        normalized = state.scaler.transform(feature_data)

        last_sequence = normalized[-sequence_length:]
        sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(state.device)

        # Generate predictions
        if request.with_uncertainty:
            mean_preds, std_preds = generate_forecast_with_uncertainty(
                state.model,
                sequence_tensor,
                request.horizon,
                request.n_samples,
                state.device,
            )

            z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(request.confidence_level, 1.96)
            lower = mean_preds - z_score * std_preds
            upper = mean_preds + z_score * std_preds

            # Inverse transform — model outputs 5 features, Close is at index 3
            # Create dummy arrays with the right shape for inverse_transform
            n_features = state.scaler.n_features_in_
            close_idx = 3  # Close column index in OHLCV

            def inverse_close(values):
                dummy = np.zeros((len(values), n_features))
                dummy[:, close_idx] = values
                return state.scaler.inverse_transform(dummy)[:, close_idx]

            predictions_real = inverse_close(mean_preds)
            lower_real = inverse_close(lower)
            upper_real = inverse_close(upper)
        else:
            state.model.eval()
            sequence = sequence_tensor.clone()
            predictions = []

            for _ in range(request.horizon):
                with torch.no_grad():
                    pred = state.model(sequence)
                predictions.append(pred.cpu().numpy().flatten())
                new_input = pred.unsqueeze(0)
                sequence = torch.cat([sequence[:, 1:, :], new_input], dim=1)

            predictions = np.array(predictions)

            # Inverse transform — extract Close (index 3) from multi-feature output
            n_features = state.scaler.n_features_in_
            close_idx = 3  # Close column index in OHLCV

            # predictions shape: (horizon, n_features)
            if predictions.ndim == 1:
                dummy = np.zeros((len(predictions), n_features))
                dummy[:, close_idx] = predictions
            else:
                dummy = np.zeros((predictions.shape[0], n_features))
                dummy[:, : predictions.shape[1]] = predictions

            predictions_real = state.scaler.inverse_transform(dummy)[:, close_idx]
            lower_real = None
            upper_real = None

        # Generate dates
        last_date = df["date"].iloc[-1]
        forecast_dates = get_forecast_dates(last_date, request.horizon)

        # Build response
        prediction_items = []
        for i, date in enumerate(forecast_dates):
            item = PredictionItem(
                date=date,
                predicted_price=float(predictions_real[i]),
                lower_bound=float(lower_real[i]) if lower_real is not None else None,
                upper_bound=float(upper_real[i]) if upper_real is not None else None,
            )
            prediction_items.append(item)

        track_prediction(success=True, duration=_time.time() - _start)
        return PredictResponse(
            predictions=prediction_items,
            last_known_price=float(df["Close"].iloc[-1]),
            last_known_date=df["date"].iloc[-1].to_pydatetime(),
            forecast_horizon=request.horizon,
            model_info=state.model_config,
            generated_at=datetime.now(),
        )

    except Exception as e:
        track_prediction(success=False, duration=_time.time() - _start)
        PREDICTION_ERRORS.labels(error_type=type(e).__name__).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get("/backtest")
async def backtest(
    days: int = 60,
    state: ModelState = Depends(get_model_state),
):
    """
    Run model on recent historical data and return actual vs predicted prices.

    Useful for visualising how well the model fits known data.
    """
    if not state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    try:
        df = load_data_from_db(start_year=2017)

        rename_map = {col: col.capitalize() for col in df.columns if col.islower()}
        if rename_map:
            df = df.rename(columns=rename_map)

        df["date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("date").reset_index(drop=True)

        sequence_length = state.model_config.get("sequence_length", 60)
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_features = [col for col in feature_columns if col in df.columns]
        feature_data = df[available_features].values
        normalized = state.scaler.transform(feature_data)

        n_features = state.scaler.n_features_in_
        close_idx = 3  # Close column index in OHLCV

        # We need at least sequence_length + days rows
        total_needed = sequence_length + days
        if len(normalized) < total_needed:
            days = len(normalized) - sequence_length

        start_idx = len(normalized) - days - sequence_length

        state.model.eval()
        results = []

        for i in range(days):
            idx = start_idx + i
            seq = normalized[idx : idx + sequence_length]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(state.device)

            with torch.no_grad():
                pred = state.model(seq_tensor)

            pred_np = pred.cpu().numpy().flatten()

            # Inverse-transform predicted Close
            dummy = np.zeros((1, n_features))
            dummy[0, : len(pred_np)] = pred_np
            pred_price = float(state.scaler.inverse_transform(dummy)[0, close_idx])

            actual_idx = idx + sequence_length
            actual_price = float(df["Close"].iloc[actual_idx])
            date_str = df["date"].iloc[actual_idx].strftime("%Y-%m-%d")

            results.append(
                {
                    "date": date_str,
                    "actual": round(actual_price, 2),
                    "predicted": round(pred_price, 2),
                }
            )

        return {"backtest": results, "days": len(results)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}",
        )


@router.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, state: ModelState = Depends(get_model_state)) -> InferenceResponse:
    """
    Perform inference on a custom input sequence.

    Use this endpoint when you want to provide your own price sequence
    for prediction instead of using historical data.
    """
    if not state.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        n_features = state.scaler.n_features_in_
        close_idx = 3  # Close column index in OHLCV

        # Normalize input — user provides Close prices; pad other features with zeros
        input_array = np.array(request.sequence)
        if input_array.ndim == 1:
            # Single-feature input: treat as Close prices, pad OHLCV
            padded = np.zeros((len(input_array), n_features))
            padded[:, close_idx] = input_array
            normalized = state.scaler.transform(padded)
        else:
            normalized = state.scaler.transform(input_array)

        # Prepare tensor
        sequence_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(state.device)

        # Generate predictions
        state.model.eval()
        sequence = sequence_tensor.clone()
        predictions = []

        for _ in range(request.steps):
            with torch.no_grad():
                pred = state.model(sequence)
            predictions.append(pred.cpu().numpy().flatten())
            new_input = pred.unsqueeze(0)
            sequence = torch.cat([sequence[:, 1:, :], new_input], dim=1)

        predictions = np.array(predictions)

        # Inverse transform — extract Close (index 3) from multi-feature output
        if predictions.ndim == 1:
            dummy = np.zeros((len(predictions), n_features))
            dummy[:, close_idx] = predictions
        else:
            dummy = np.zeros((predictions.shape[0], n_features))
            dummy[:, : predictions.shape[1]] = predictions

        predictions_real = state.scaler.inverse_transform(dummy)[:, close_idx]

        return InferenceResponse(
            predictions=predictions_real.tolist(),
            input_length=len(request.sequence),
            generated_at=datetime.now(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )
