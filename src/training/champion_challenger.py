"""Champion-Challenger model evaluation pipeline.

Implements automated model promotion with champion-challenger comparison:
1. Detect retrain triggers (PSI drift, staleness, CI breach)
2. Train challenger model using **Optuna HPO** on new data
3. Compare challenger vs champion on holdout set
4. Only promote if challenger significantly outperforms champion

The challenger uses Bayesian hyperparameter optimization (Optuna TPE sampler)
to search for the best configuration, giving it a fair chance to beat the
champion even when data dynamics have changed.

Thresholds:
    - δ RMSE ≤ -0.5% → promote challenger as new champion
    - δ RMSE > 0      → keep champion (challenger is worse)

References:
    - MLflow Model Registry: https://mlflow.org/docs/latest/model-registry.html
    - Sato, Wider, Windheuser (2019) — Continuous Delivery for ML
    - Akiba et al. (2019) — Optuna: A Next-generation Hyperparameter Optimization Framework
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "champion_challenger"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optuna persisted storage for dashboard visibility
OPTUNA_DB_PATH = ROOT_DIR / "outputs" / "optuna_studies.db"
OPTUNA_STORAGE_URL = f"sqlite:///{OPTUNA_DB_PATH}"

# Promotion threshold: challenger must improve RMSE by at least this fraction
IMPROVEMENT_THRESHOLD = 0.005  # 0.5%


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "directional_accuracy": self.directional_accuracy,
            "timestamp": self.timestamp,
        }


@dataclass
class ComparisonResult:
    """Result of a champion-challenger comparison."""

    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    promote: bool = False
    reason: str = ""
    rmse_delta: float = 0.0
    rmse_delta_pct: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "champion": self.champion_metrics.to_dict(),
            "challenger": self.challenger_metrics.to_dict(),
            "promote": self.promote,
            "reason": self.reason,
            "rmse_delta": round(self.rmse_delta, 6),
            "rmse_delta_pct": round(self.rmse_delta_pct, 4),
            "timestamp": self.timestamp,
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scaler: Any,
    device: str = "cpu",
    target_idx: int = 0,
) -> ModelMetrics:
    """Evaluate a model on a dataset and return comprehensive metrics.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the evaluation dataset.
        scaler: Fitted scaler for inverse transformation.
        device: Device for inference.
        target_idx: Index of the target column for inverse transform.

    Returns:
        ModelMetrics with RMSE, MAE, MAPE, R², directional accuracy.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)

            # For multi-output models, extract only the target column
            if preds.dim() == 2 and preds.shape[1] > 1:
                preds = preds[:, target_idx : target_idx + 1]

            # Align targets to the same column as predictions
            if y_batch.dim() == 2 and y_batch.shape[1] > 1:
                y_batch = y_batch[:, target_idx : target_idx + 1]

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Inverse transform if scaler available
    if scaler is not None:
        n_features = getattr(scaler, "n_features_in_", 1)
        if n_features > 1:
            dummy_pred = np.zeros((len(predictions), n_features))
            dummy_pred[:, target_idx] = predictions
            predictions = scaler.inverse_transform(dummy_pred)[:, target_idx]

            dummy_tgt = np.zeros((len(targets), n_features))
            dummy_tgt[:, target_idx] = targets
            targets = scaler.inverse_transform(dummy_tgt)[:, target_idx]
        else:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Compute metrics
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    mae = float(np.mean(np.abs(predictions - targets)))

    # MAPE (avoid division by zero)
    mask = np.abs(targets) > 1e-8
    mape = float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100) if mask.any() else 0.0

    # R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy
    if len(targets) > 1:
        actual_dir = np.sign(np.diff(targets))
        pred_dir = np.sign(np.diff(predictions))
        dir_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_accuracy = 0.0

    return ModelMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        r2=r2,
        directional_accuracy=dir_accuracy,
    )


def compare_models(
    champion_metrics: ModelMetrics,
    challenger_metrics: ModelMetrics,
    threshold: float = IMPROVEMENT_THRESHOLD,
) -> ComparisonResult:
    """Compare champion and challenger models.

    Promotion criteria:
        Challenger is promoted if its RMSE is at least `threshold` (0.5%)
        better than the champion.

    Args:
        champion_metrics: Metrics of the current production model.
        challenger_metrics: Metrics of the newly trained model.
        threshold: Minimum relative improvement to promote.

    Returns:
        ComparisonResult with promotion decision and reasoning.
    """
    rmse_delta = challenger_metrics.rmse - champion_metrics.rmse
    rmse_delta_pct = rmse_delta / champion_metrics.rmse if champion_metrics.rmse > 0 else 0.0

    result = ComparisonResult(
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        rmse_delta=rmse_delta,
        rmse_delta_pct=rmse_delta_pct,
    )

    if rmse_delta_pct <= -threshold:
        result.promote = True
        result.reason = (
            f"RMSE do Challenger ({challenger_metrics.rmse:.4f}) é "
            f"{abs(rmse_delta_pct) * 100:.2f}% melhor que o Champion "
            f"({champion_metrics.rmse:.4f}). Promovendo."
        )
        logger.info("✅ PROMOTE: %s", result.reason)
    elif rmse_delta < 0:
        result.promote = False
        result.reason = (
            f"RMSE do Challenger melhorou {abs(rmse_delta_pct) * 100:.2f}% "
            f"mas abaixo do threshold ({threshold * 100:.1f}%). Mantendo Champion."
        )
        logger.info("⚠️ NO PROMOTE (below threshold): %s", result.reason)
    else:
        result.promote = False
        result.reason = (
            f"RMSE do Challenger ({challenger_metrics.rmse:.4f}) é pior "
            f"que o Champion ({champion_metrics.rmse:.4f}). Mantendo Champion."
        )
        logger.info("❌ NO PROMOTE: %s", result.reason)

    return result


def _train_challenger(
    experiment_name: str,
    champion_path: Optional[str] = None,
    n_trials: int = 20,
) -> dict:
    """Train a challenger model using Optuna HPO and compare with champion.

    Instead of retraining with fixed hyperparameters, the challenger runs
    a Bayesian hyperparameter search (Optuna TPE sampler) to find the best
    configuration for the current data distribution. This gives the
    challenger a fair advantage to beat the champion when data has shifted.

    Pipeline:
        1. Load & prepare data
        2. Evaluate champion on validation set
        3. Run Optuna HPO (n_trials) to find best hyperparameters
        4. Train final challenger with best params (full epochs)
        5. Return both losses for comparison

    Args:
        experiment_name: MLflow experiment name.
        champion_path: Path to champion checkpoint. Defaults to best_model.pth.
        n_trials: Number of Optuna trials for HPO (default 20 for speed).

    Returns:
        dict with ``run_id``, ``best_val_loss`` (challenger),
        ``champion_val_loss``, and ``optuna_best_params``.
    """
    import mlflow
    import optuna
    import torch
    import torch.nn as nn

    from src.config import (
        BIDIRECTIONAL,
        DATABASE_PATH,
        DROPOUT,
        EARLY_STOPPING_PATIENCE,
        EPOCHS,
        HIDDEN_SIZE,
        NUM_LAYERS,
        SEQUENCE_LENGTH,
        TEST_SPLIT,
        TRAIN_SPLIT,
        VAL_SPLIT,
    )
    from src.data.preprocessing import (
        create_sequences,
        load_data_from_db,
        normalize_features,
        train_val_test_split,
    )
    from src.models.lstm_model import create_model
    from src.training.train import train_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Data preparation ──────────────────────────────────────────────────
    df = load_data_from_db(str(DATABASE_PATH))
    feature_columns = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    normalized_data, scaler = normalize_features(df, feature_columns)
    X, y = create_sequences(normalized_data, sequence_length=SEQUENCE_LENGTH)
    X_train, y_train, X_val, y_val, _X_test, _y_test = train_val_test_split(
        X, y, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT, test_ratio=TEST_SPLIT
    )

    input_size = X.shape[2]
    output_size = y.shape[1]

    # ── Evaluate champion ────────────────────────────────────────────────
    champion_path_resolved = Path(champion_path) if champion_path else ROOT_DIR / "models" / "best_model.pth"
    champion_val_loss = float("inf")
    if champion_path_resolved.exists():
        try:
            champ_model = create_model(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                bidirectional=BIDIRECTIONAL,
                output_size=output_size,
            )
            ckpt = torch.load(str(champion_path_resolved), map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            champ_model.load_state_dict(state, strict=False)
            champ_model.to(device).eval()

            criterion = nn.MSELoss()
            with torch.no_grad():
                preds = champ_model(torch.FloatTensor(X_val).to(device))
                champion_val_loss = criterion(preds, torch.FloatTensor(y_val).to(device)).item()
            logger.info("Champion val loss: %.6f", champion_val_loss)
        except Exception as exc:
            logger.warning("Could not evaluate champion model: %s — using inf", exc)
    else:
        logger.warning("Champion checkpoint not found at %s — using inf", champion_path_resolved)

    # ── Optuna HPO for challenger ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running Optuna HPO with %d trials for challenger", n_trials)
    logger.info("=" * 60)

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _optuna_objective(trial: optuna.Trial) -> float:
        """Optuna objective: train LSTM with suggested params, return val loss."""
        # Suggest hyperparameters
        hp_hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        hp_num_layers = trial.suggest_int("num_layers", 1, 4)
        hp_learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        hp_dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hp_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        model = create_model(
            input_size=input_size,
            hidden_size=hp_hidden_size,
            num_layers=hp_num_layers,
            dropout=hp_dropout,
            bidirectional=BIDIRECTIONAL,
            output_size=output_size,
        ).to(device)

        config = {
            "batch_size": hp_batch_size,
            "learning_rate": hp_learning_rate,
            "epochs": min(EPOCHS, 30),  # Reduced epochs for HPO speed
            "early_stopping_patience": 5,
            "optimizer": "Adam",
        }

        try:
            _trained, history = train_model(
                model=model,
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                config=config,
                device=device,
                mlflow_tracking=False,  # Don't clutter MLflow during HPO
            )
            best_val_loss = min(history["val_loss"])
            return best_val_loss
        except Exception as e:
            logger.warning("Optuna trial %d failed: %s", trial.number, e)
            raise optuna.TrialPruned(str(e))

    study = optuna.create_study(
        study_name=f"challenger_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=OPTUNA_STORAGE_URL,
        load_if_exists=False,
    )
    study.optimize(_optuna_objective, n_trials=n_trials)

    best_params = study.best_params
    best_hpo_loss = study.best_value
    logger.info("Optuna best val loss: %.6f", best_hpo_loss)
    logger.info("Optuna best params: %s", best_params)

    # ── Train final challenger with best params (full epochs) ────────────
    logger.info("Training final challenger with Optuna best params (full epochs)")

    challenger_model = create_model(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        bidirectional=BIDIRECTIONAL,
        output_size=output_size,
    ).to(device)

    final_config = {
        "batch_size": best_params["batch_size"],
        "learning_rate": best_params["learning_rate"],
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "optimizer": "Adam",
    }

    mlflow.set_tracking_uri(str(ROOT_DIR / "mlruns"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="challenger_optuna_training") as run:
        # Log Optuna HPO metadata
        mlflow.log_params({f"optuna_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("optuna_n_trials", n_trials)
        mlflow.log_metric("optuna_best_trial_loss", best_hpo_loss)
        mlflow.set_tag("training_method", "optuna_hpo")
        mlflow.set_tag("sampler", "TPE")
        mlflow.set_tag("n_completed_trials", len(study.trials))

        _trained, history = train_model(
            model=challenger_model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            config=final_config,
            device=device,
            mlflow_tracking=True,
        )
        run_id = run.info.run_id

    challenger_loss = min(history["val_loss"])
    logger.info("Challenger final val loss: %.6f (HPO trial best: %.6f)", challenger_loss, best_hpo_loss)

    # Build champion model reference for full evaluation
    champ_model_ref = None
    if champion_path_resolved.exists():
        try:
            champ_model_ref = create_model(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                bidirectional=BIDIRECTIONAL,
                output_size=output_size,
            )
            ckpt = torch.load(str(champion_path_resolved), map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            champ_model_ref.load_state_dict(state, strict=False)
            champ_model_ref.to(device).eval()
        except Exception as exc:
            logger.warning("Could not reload champion for full eval: %s", exc)
            champ_model_ref = None

    # Determine target_idx (Close column)
    close_idx = feature_columns.index("Close") if "Close" in feature_columns else 3

    return {
        "run_id": run_id,
        "best_val_loss": challenger_loss,
        "champion_val_loss": champion_val_loss,
        "optuna_best_params": best_params,
        "optuna_n_trials": n_trials,
        "optuna_best_trial_loss": best_hpo_loss,
        "_challenger_model": _trained,
        "_champion_model": champ_model_ref,
        "_test_data": (_X_test, _y_test),
        "_scaler": scaler,
        "_target_idx": close_idx,
        "_device": str(device),
    }


def run_champion_challenger(
    champion_path: Optional[str] = None,
    experiment_name: str = "champion_challenger",
    retrain_on_drift: bool = True,
) -> dict:
    """Run the full champion-challenger pipeline.

    Steps:
        1. Check for retrain triggers (PSI drift, staleness, CI breach)
        2. If any trigger fires (or forced), train challenger
        3. Evaluate both on holdout set
        4. Compare and decide promotion
        5. Log results to MLflow

    Args:
        champion_path: Path to champion model checkpoint. Defaults to best_model.pth.
        experiment_name: MLflow experiment name.
        retrain_on_drift: If True, only retrain when at least one trigger fires.

    Returns:
        Dictionary with pipeline results.
    """
    from src.monitoring.drift import detect_all_triggers

    result = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": False,
        "retrained": False,
        "comparison": None,
        "promoted": False,
        "active_triggers": [],
    }

    # Step 0: Refresh stock database with latest market data
    logger.info("=" * 60)
    logger.info("Step 0: Refreshing Stock Database (ETL)")
    logger.info("=" * 60)
    try:
        from src.etl import refresh_stock_data

        refresh_stock_data()
    except Exception as e:
        logger.warning("ETL refresh failed (continuing with existing data): %s", e)

    # Step 1: Multi-trigger detection
    logger.info("=" * 60)
    logger.info("Step 1: Multi-Trigger Retrain Detection")
    logger.info("=" * 60)

    trigger_report = detect_all_triggers(model_path=champion_path)
    retrain_recommended = trigger_report.get("retrain_recommended", False)
    active_triggers = trigger_report.get("active_triggers", [])

    result["drift_detected"] = retrain_recommended
    result["active_triggers"] = active_triggers
    result["trigger_report"] = trigger_report

    if retrain_on_drift and not retrain_recommended:
        logger.info("No retrain triggers active. Skipping retraining.")
        result["reason"] = "No retrain triggers active"
        _save_result(result)
        return result

    logger.info(
        "Retrain triggered by: %s",
        ", ".join(active_triggers) if active_triggers else "forced",
    )

    # Step 2: Train challenger
    logger.info("=" * 60)
    logger.info("Step 2: Training Challenger Model")
    logger.info("=" * 60)

    try:
        training_result = _train_challenger(experiment_name, champion_path=champion_path)
        result["retrained"] = True
        result["training_result"] = {
            "run_id": training_result.get("run_id"),
            "best_val_loss": training_result.get("best_val_loss"),
            "optuna_best_params": training_result.get("optuna_best_params"),
            "optuna_n_trials": training_result.get("optuna_n_trials"),
            "optuna_best_trial_loss": training_result.get("optuna_best_trial_loss"),
        }
    except Exception as e:
        logger.error("Challenger training failed: %s", str(e))
        result["error"] = str(e)
        _save_result(result)
        return result

    # Step 3 & 4: Compare
    logger.info("=" * 60)
    logger.info("Step 3: Champion-Challenger Comparison")
    logger.info("=" * 60)

    # ── Full evaluation on test set with all metrics ──────────────────
    from torch.utils.data import DataLoader, TensorDataset

    _challenger_model = training_result.get("_challenger_model")
    _champion_model = training_result.get("_champion_model")
    _test_data = training_result.get("_test_data")
    _scaler = training_result.get("_scaler")
    _target_idx = training_result.get("_target_idx", 3)
    _device = training_result.get("_device", "cpu")

    if _challenger_model is not None and _test_data is not None and _scaler is not None:
        X_test, y_test = _test_data
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        challenger_m = evaluate_model(_challenger_model, test_loader, _scaler, device=_device, target_idx=_target_idx)
        logger.info("Challenger full metrics: %s", challenger_m.to_dict())

        if _champion_model is not None:
            champion_m = evaluate_model(_champion_model, test_loader, _scaler, device=_device, target_idx=_target_idx)
            logger.info("Champion full metrics: %s", champion_m.to_dict())
        else:
            champion_loss = training_result.get("champion_val_loss", float("inf"))
            champion_m = ModelMetrics(rmse=champion_loss)
    else:
        # Fallback: only RMSE from val_loss (legacy behavior)
        champion_loss = training_result.get("champion_val_loss", float("inf"))
        challenger_loss = training_result.get("best_val_loss", float("inf"))
        champion_m = ModelMetrics(rmse=champion_loss)
        challenger_m = ModelMetrics(rmse=challenger_loss)

    comparison = compare_models(champion_m, challenger_m)
    result["comparison"] = comparison.to_dict()
    result["promoted"] = comparison.promote

    # Step 4b: If promoted, save challenger as new best_model.pth
    if comparison.promote:
        logger.info("🏆 Promoting challenger → saving to models/best_model.pth")
        try:
            import pickle as _pickle

            import torch as _torch

            _challenger_model = training_result.get("_challenger_model")
            _scaler = training_result.get("_scaler")
            _best_params = training_result.get("optuna_best_params") or {}

            if _challenger_model is not None:
                model_path = ROOT_DIR / "models" / "best_model.pth"
                scaler_path = ROOT_DIR / "models" / "scaler.pkl"
                model_path.parent.mkdir(parents=True, exist_ok=True)

                # Build model_config so API can reconstruct the architecture
                model_config = {
                    "input_size": 5,
                    "hidden_size": _best_params.get("hidden_size", 128),
                    "num_layers": _best_params.get("num_layers", 2),
                    "output_size": 5,
                    "dropout": _best_params.get("dropout", 0.2),
                    "bidirectional": False,
                }

                _torch.save(
                    {
                        "model_state_dict": _challenger_model.state_dict(),
                        "model_config": model_config,
                        "promoted_from": "champion_challenger",
                        "promoted_at": datetime.now().isoformat(),
                        "optuna_best_params": _best_params,
                        "run_id": training_result.get("run_id"),
                        "challenger_rmse": challenger_m.rmse,
                        "champion_rmse": champion_m.rmse,
                        "rmse_improvement_pct": abs(comparison.rmse_delta_pct) * 100,
                    },
                    str(model_path),
                )
                logger.info("✅ New champion saved to %s", model_path)

                # Save updated scaler so API predictions are consistent
                if _scaler is not None:
                    with open(scaler_path, "wb") as f:
                        _pickle.dump(_scaler, f)
                    logger.info("✅ Updated scaler saved to %s", scaler_path)
            else:
                logger.warning("Challenger model object not available — skipping save")
        except Exception as e:
            logger.error("Failed to save promoted model: %s", e)

    # Step 5: Log to MLflow
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="champion_challenger_evaluation"):
            mlflow.log_params(
                {
                    "drift_detected": result["drift_detected"],
                    "retrained": result["retrained"],
                    "promoted": comparison.promote,
                    "active_triggers": ", ".join(active_triggers) if active_triggers else "forced",
                }
            )
            mlflow.log_metrics(
                {
                    "champion_rmse": champion_m.rmse,
                    "challenger_rmse": challenger_m.rmse,
                    "rmse_delta": comparison.rmse_delta,
                    "rmse_delta_pct": comparison.rmse_delta_pct,
                    "n_active_triggers": len(active_triggers),
                    "optuna_n_trials": training_result.get("optuna_n_trials", 0),
                }
            )
            mlflow.set_tag("pipeline", "champion_challenger")
            mlflow.set_tag("promotion_decision", "promote" if comparison.promote else "keep_champion")
            mlflow.set_tag("trigger_types", ", ".join(active_triggers) if active_triggers else "none")
            mlflow.set_tag("challenger_method", "optuna_hpo")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", str(e))

    _save_result(result)
    return result


def _save_result(result: dict) -> None:
    """Save pipeline result to JSON."""
    output_path = RESULTS_DIR / "latest_comparison.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)
