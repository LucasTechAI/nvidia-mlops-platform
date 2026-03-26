"""
Model Metrics Component for Dashboard.

Displays model performance metrics and training history.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

project_root = Path(__file__).resolve().parent.parent.parent.parent


def load_checkpoint_info() -> dict:
    """Load information from the best model checkpoint."""
    # Search multiple possible locations
    candidates = [
        project_root / "models" / "best_model.pth",
        project_root / "models" / "best_model.pt",
        project_root / "data" / "models" / "checkpoints" / "best_model.pt",
        project_root / "data" / "models" / "checkpoints" / "best_model.pth",
    ]

    checkpoint_path = None
    for path in candidates:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        return None

    try:
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle bare state_dict (OrderedDict of tensors) vs checkpoint dict
        if isinstance(data, dict) and "model_state_dict" not in data and all(
            isinstance(v, torch.Tensor) for v in list(data.values())[:3]
        ):
            # Bare state_dict — wrap into expected checkpoint format
            state_dict = data
            input_size = state_dict["lstm.weight_ih_l0"].shape[1] if "lstm.weight_ih_l0" in state_dict else 5
            hidden_size = state_dict["lstm.weight_hh_l0"].shape[1] if "lstm.weight_hh_l0" in state_dict else 128
            output_size = state_dict["fc.bias"].shape[0] if "fc.bias" in state_dict else 1
            num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l")) or 2

            # Try to enrich with MLflow data
            mlflow_metrics = load_mlflow_metrics()
            mlflow_params = load_mlflow_params()

            best_val_loss = 0.0
            best_epoch = 0
            total_epochs = 0
            early_stopped = False
            test_results = {}

            if not mlflow_metrics.empty:
                # Extract best_val_loss scalar
                bvl = mlflow_metrics[mlflow_metrics["key"] == "best_val_loss"]
                if not bvl.empty:
                    best_val_loss = bvl["value"].iloc[0]

                # Find best epoch from val_loss
                val_loss = mlflow_metrics[mlflow_metrics["key"] == "val_loss"]
                if not val_loss.empty:
                    best_idx = val_loss["value"].idxmin()
                    best_epoch = int(val_loss.loc[best_idx, "step"])
                    total_epochs = int(val_loss["step"].max()) + 1

                # Check early stopping: configured epochs vs actual
                configured_epochs = int(mlflow_params.get("epochs", total_epochs))
                if total_epochs < configured_epochs:
                    early_stopped = True

                # Use last-step validation metrics as test results
                for metric_key, result_key in [
                    ("val_rmse", "rmse"),
                    ("val_mae", "mae"),
                    ("val_mape", "mape"),
                ]:
                    m = mlflow_metrics[mlflow_metrics["key"] == metric_key]
                    if not m.empty:
                        # Value at best epoch
                        at_best = m[m["step"] == best_epoch]
                        if not at_best.empty:
                            test_results[result_key] = at_best["value"].iloc[0]
                        else:
                            test_results[result_key] = m.iloc[-1]["value"]

            data = {
                "model_state_dict": state_dict,
                "model_config": {
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "output_size": output_size,
                    "num_layers": num_layers,
                },
                "epoch": best_epoch,
                "loss": best_val_loss,
                "training_info": {
                    "Best Epoch": best_epoch,
                    "Best Val Loss": best_val_loss,
                    "Total Epochs": total_epochs,
                    "Early Stopped": early_stopped,
                },
                "test_results": test_results,
            }

        return data
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None


def _find_latest_mlflow_run() -> Path | None:
    """Find the latest MLflow run directory (file store)."""
    # Search in both mlruns/ and data/mlruns/
    candidates = [
        project_root / "mlruns",
        project_root / "data" / "mlruns",
    ]

    latest_run = None
    latest_time = 0

    for mlruns_root in candidates:
        if not mlruns_root.exists():
            continue
        for exp_dir in mlruns_root.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "models":
                continue
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name == "models":
                    continue
                meta_file = run_dir / "meta.yaml"
                metrics_dir = run_dir / "metrics"
                if meta_file.exists() and metrics_dir.exists():
                    mtime = meta_file.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_run = run_dir

    return latest_run


def load_mlflow_metrics() -> pd.DataFrame:
    """Load metrics from MLflow file store."""
    run_dir = _find_latest_mlflow_run()
    if run_dir is None:
        return pd.DataFrame()

    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return pd.DataFrame()

    try:
        rows = []
        for metric_file in metrics_dir.iterdir():
            if not metric_file.is_file():
                continue
            key = metric_file.name
            with open(metric_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) >= 3:
                        timestamp, value, step = parts[0], parts[1], parts[2]
                        rows.append({
                            "key": key,
                            "value": float(value),
                            "step": int(step),
                            "timestamp": int(timestamp),
                        })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["key", "step"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading MLflow metrics: {e}")
        return pd.DataFrame()


def load_mlflow_params() -> dict:
    """Load parameters from the latest MLflow run."""
    run_dir = _find_latest_mlflow_run()
    if run_dir is None:
        return {}
    params_dir = run_dir / "params"
    if not params_dir.exists():
        return {}
    params = {}
    for pf in params_dir.iterdir():
        if pf.is_file():
            params[pf.name] = pf.read_text().strip()
    return params


def load_hpo_results() -> dict:
    """Load HPO best parameters."""
    hpo_paths = list((project_root / "data" / "outputs" / "hpo").glob("*/best_params.json"))

    if not hpo_paths:
        return None

    # Get the most recent one
    latest = max(hpo_paths, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest, "r") as f:
            return json.load(f)
    except Exception:
        return None


def render_metrics_page():
    """Render the model metrics page."""

    # Enhanced page header
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-weight: 600;">📈 Model Performance Metrics</h2>
            <p style="color: rgba(250,250,250,0.6); margin-top: 0.5rem;">
                Explore training history, evaluation metrics, and performance analysis of the LSTM model.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Load data
    with st.spinner("🔄 Loading model metrics..."):
        checkpoint = load_checkpoint_info()
        mlflow_metrics = load_mlflow_metrics()
        hpo_params = load_hpo_results()

    if checkpoint is None:
        st.markdown(
            """
            <div style="
                background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
                border: 1px solid rgba(255, 193, 7, 0.3);
                border-radius: 12px;
                padding: 24px;
                text-align: center;
            ">
                <span style="font-size: 2.5rem;">🤖</span>
                <h3 style="margin: 0.5rem 0; color: #FFC107;">No Model Found</h3>
                <p style="color: rgba(250,250,250,0.7); margin: 0;">
                    Please train the model first to view performance metrics.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )
        return

    # Training Overview Section
    st.markdown(
        """
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            🎯 Training Overview
        </p>
    """,
        unsafe_allow_html=True,
    )

    training_info = checkpoint.get("training_info", {})
    _model_config = checkpoint.get("model_config", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Best Epoch",
            value=training_info.get("Best Epoch", checkpoint.get("epoch", "N/A")),
        )

    with col2:
        best_loss = training_info.get("Best Val Loss", checkpoint.get("loss", 0))
        st.metric(
            label="Best Validation Loss",
            value=f"{best_loss:.6f}" if isinstance(best_loss, (int, float)) else str(best_loss),
        )

    with col3:
        total_epochs = training_info.get("Total Epochs", checkpoint.get("epoch", "N/A"))
        st.metric(label="Total Epochs Trained", value=total_epochs)

    with col4:
        early_stop = training_info.get("Early Stopped", False)
        st.metric(label="Early Stopping", value="Yes ✅" if early_stop else "No ❌")

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance Metrics Section
    # Determine if these are true test metrics or just validation approximations
    test_results = checkpoint.get("test_results", {})
    has_full_test = any(
        test_results.get(k, 0) != 0
        for k in ["r2_score", "correlation", "directional_accuracy"]
    )
    section_title = "📊 Test Set Performance" if has_full_test else "📊 Validation Performance (Best Epoch)"

    st.markdown(
        f"""
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            {section_title}
        </p>
    """,
        unsafe_allow_html=True,
    )

    if test_results:
        # --- Primary regression metrics (always available) ---
        primary = []
        rmse = test_results.get("rmse", test_results.get("RMSE"))
        if rmse is not None:
            primary.append(("RMSE", f"${rmse:.4f}"))
        mae = test_results.get("mae", test_results.get("MAE"))
        if mae is not None:
            primary.append(("MAE", f"${mae:.4f}"))
        mape = test_results.get("mape", test_results.get("MAPE"))
        if mape is not None:
            primary.append(("MAPE", f"{mape:.2f}%"))
        r2 = test_results.get("r2_score", test_results.get("R2 Score"))
        if r2 is not None and has_full_test:
            quality = "🟢 Excellent" if r2 > 0.9 else ("🟡 Good" if r2 > 0.7 else "🔴 Needs Work")
            primary.append(("R² Score", f"{r2:.4f}", quality))

        cols = st.columns(len(primary)) if primary else []
        for i, item in enumerate(primary):
            with cols[i]:
                if len(item) == 3:
                    st.metric(label=item[0], value=item[1], delta=item[2])
                else:
                    st.metric(label=item[0], value=item[1])

        # --- Secondary metrics (only if computed during test evaluation) ---
        secondary = []
        corr = test_results.get("correlation", test_results.get("Correlation"))
        if corr is not None and corr != 0:
            secondary.append(("Correlation", f"{corr:.4f}"))
        dir_acc = test_results.get("directional_accuracy", test_results.get("Directional Accuracy"))
        if dir_acc is not None and dir_acc != 0:
            secondary.append(("Directional Accuracy", f"{dir_acc:.1f}%"))
        sharpe = test_results.get("sharpe_ratio", test_results.get("Sharpe Ratio"))
        if sharpe is not None and sharpe != 0:
            secondary.append(("Sharpe Ratio", f"{sharpe:.2f}"))
        max_dd = test_results.get("max_drawdown", test_results.get("Max Drawdown"))
        if max_dd is not None and max_dd != 0:
            secondary.append(("Max Drawdown", f"{max_dd:.1f}%"))

        if secondary:
            st.markdown("<br>", unsafe_allow_html=True)
            cols2 = st.columns(len(secondary))
            for i, (label, value) in enumerate(secondary):
                with cols2[i]:
                    st.metric(label=label, value=value)
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(145deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%);
                border: 1px solid rgba(33, 150, 243, 0.3);
                border-radius: 10px;
                padding: 16px;
            ">
                <span style="color: #2196F3;">ℹ️</span> No metrics available. Please train the model first.
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Training History Visualization
    st.markdown(
        """
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📉 Training History
        </p>
    """,
        unsafe_allow_html=True,
    )

    if not mlflow_metrics.empty:
        render_training_curves(mlflow_metrics)
    else:
        # Try to get from checkpoint
        if "train_losses" in checkpoint or "val_losses" in checkpoint:
            render_training_curves_from_checkpoint(checkpoint)
        else:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(145deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%);
                    border: 1px solid rgba(33, 150, 243, 0.3);
                    border-radius: 10px;
                    padding: 16px;
                ">
                    <span style="color: #2196F3;">ℹ️</span> Training history not available.
                </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Metrics Interpretation
    st.markdown(
        """
        <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            📝 Metrics Interpretation
        </p>
    """,
        unsafe_allow_html=True,
    )

    render_metrics_interpretation(test_results)

    # HPO Results
    if hpo_params:
        st.markdown("---")
        st.markdown(
            """
            <p style="color: rgba(250,250,250,0.5); font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
                🔬 Best Hyperparameters (from HPO)
            </p>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            params_df = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in hpo_params.items()])
            st.dataframe(params_df, width="stretch", hide_index=True)

        with col2:
            # Radar chart of normalized parameters
            render_hpo_radar(hpo_params)


def render_training_curves(metrics_df: pd.DataFrame):
    """Render training curves from MLflow metrics."""

    # Filter for training metrics
    loss_metrics = metrics_df[metrics_df["key"].str.contains("loss", case=False)]

    if loss_metrics.empty:
        st.info("No loss metrics found in MLflow.")
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Loss", "RMSE", "MAE", "R² Score"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    metric_configs = [
        ("loss", 1, 1, "#FF6B6B"),
        ("rmse", 1, 2, "#4ECDC4"),
        ("mae", 2, 1, "#45B7D1"),
        ("r2", 2, 2, "#96CEB4"),
    ]

    for metric_key, row, col, color in metric_configs:
        # Get train and val metrics
        train_data = metrics_df[
            (metrics_df["key"].str.contains(f"train_{metric_key}", case=False))
            | (
                metrics_df["key"].str.contains(f"{metric_key}", case=False)
                & ~metrics_df["key"].str.contains("val|test", case=False)
            )
        ]
        val_data = metrics_df[metrics_df["key"].str.contains(f"val_{metric_key}", case=False)]

        if not train_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=train_data["step"],
                    y=train_data["value"],
                    mode="lines",
                    name=f"Train {metric_key.upper()}",
                    line=dict(color=color),
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )

        if not val_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=val_data["step"],
                    y=val_data["value"],
                    mode="lines",
                    name=f"Val {metric_key.upper()}",
                    line=dict(color=color, dash="dash"),
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        title=dict(text="Training Metrics Over Epochs", font=dict(size=18)),
    )

    st.plotly_chart(fig, width="stretch")


def render_training_curves_from_checkpoint(checkpoint: dict):
    """Render training curves from checkpoint data."""

    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])

    if not train_losses and not val_losses:
        st.info("No training history in checkpoint.")
        return

    fig = go.Figure()

    if train_losses:
        fig.add_trace(
            go.Scatter(
                y=train_losses,
                mode="lines",
                name="Training Loss",
                line=dict(color="#76B900", width=2),
            )
        )

    if val_losses:
        fig.add_trace(
            go.Scatter(
                y=val_losses,
                mode="lines",
                name="Validation Loss",
                line=dict(color="#FF6B35", width=2),
            )
        )

    fig.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss (MSE)",
        template="plotly_dark",
        height=400,
    )

    st.plotly_chart(fig, width="stretch")


def render_metrics_interpretation(test_results: dict):
    """Render interpretation of metrics."""

    interpretations = []

    if test_results:
        r2 = test_results.get("r2_score", test_results.get("R2 Score"))
        if r2 is not None and isinstance(r2, (int, float)) and r2 != 0:
            if r2 > 0.95:
                interpretations.append(
                    (
                        "R² Score",
                        "🟢 Excellent",
                        f"The model explains {r2 * 100:.1f}% of the variance in stock prices.",
                    )
                )
            elif r2 > 0.85:
                interpretations.append(
                    (
                        "R² Score",
                        "🟡 Good",
                        f"The model explains {r2 * 100:.1f}% of the variance. Room for improvement.",
                    )
                )
            else:
                interpretations.append(
                    (
                        "R² Score",
                        "🔴 Needs Improvement",
                        f"Only {r2 * 100:.1f}% variance explained.",
                    )
                )

        mape = test_results.get("mape", test_results.get("MAPE"))
        if mape is not None and isinstance(mape, (int, float)) and mape != 0:
            if mape < 5:
                interpretations.append(
                    (
                        "MAPE",
                        "🟢 Excellent",
                        f"Average prediction error is only {mape:.2f}%.",
                    )
                )
            elif mape < 10:
                interpretations.append(
                    (
                        "MAPE",
                        "🟡 Good",
                        f"Average prediction error of {mape:.2f}% is acceptable.",
                    )
                )
            else:
                interpretations.append(
                    (
                        "MAPE",
                        "🔴 High Error",
                        f"Average prediction error of {mape:.2f}% is concerning.",
                    )
                )

        dir_acc = test_results.get("directional_accuracy", test_results.get("Directional Accuracy"))
        if dir_acc is not None and isinstance(dir_acc, (int, float)) and dir_acc != 0:
            if dir_acc > 55:
                interpretations.append(
                    (
                        "Directional Accuracy",
                        "🟢 Good",
                        f"Model correctly predicts direction {dir_acc:.1f}% of the time.",
                    )
                )
            elif dir_acc > 50:
                interpretations.append(
                    (
                        "Directional Accuracy",
                        "🟡 Marginal",
                        f"Slightly better than random at {dir_acc:.1f}%.",
                    )
                )
            else:
                interpretations.append(
                    (
                        "Directional Accuracy",
                        "🔴 Poor",
                        f"Only {dir_acc:.1f}% - worse than random guessing.",
                    )
                )

    if interpretations:
        for metric, status, description in interpretations:
            st.markdown(f"**{metric}** {status}")
            st.markdown(f"_{description}_")
            st.markdown("")
    else:
        st.info("ℹ️ Interpretation requires test metrics.")


def render_hpo_radar(params: dict):
    """Render radar chart of hyperparameters."""

    # Normalize parameters for visualization
    param_ranges = {
        "hidden_size": (32, 256),
        "num_layers": (1, 4),
        "learning_rate": (0.0001, 0.01),
        "dropout": (0.0, 0.5),
        "sequence_length": (30, 120),
        "batch_size": (16, 128),
    }

    categories = []
    values = []

    for param, (min_val, max_val) in param_ranges.items():
        if param in params:
            val = params[param]
            # Normalize to 0-1
            if param == "learning_rate":
                normalized = (np.log10(val) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val))
            else:
                normalized = (val - min_val) / (max_val - min_val)
            categories.append(param.replace("_", " ").title())
            values.append(max(0, min(1, normalized)))

    if not categories:
        return

    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(118, 185, 0, 0.3)",
            line=dict(color="#76B900", width=2),
            name="Best Parameters",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        template="plotly_dark",
        height=300,
        margin=dict(l=60, r=60, t=40, b=40),
        title=dict(text="Parameter Distribution", font=dict(size=14)),
    )

    st.plotly_chart(fig, width="stretch")
