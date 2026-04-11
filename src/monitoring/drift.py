"""Drift detection with Evidently — Multi-trigger retrain system.

Three independent retrain triggers (any one is sufficient):

1. **Data Drift (PSI)**
   - PSI > 0.1 → warning (distribution shift detected)
   - PSI > 0.2 → retrain trigger (significant drift)

2. **Model Staleness**
   - If ≥ 30 days have passed since the last model training/update,
     the model is considered stale and retraining is recommended.
   - Rationale: even without measurable drift, financial markets evolve
     and a model that hasn't been refreshed may silently degrade.
   - Reference: Sculley et al. (2015) — "Hidden Technical Debt in ML Systems"

3. **Prediction Error Breach (Concept Drift)**
   - If the fraction of actual values falling outside the model's 95%
     prediction confidence interval exceeds a threshold (default 20%),
     the model's learned patterns no longer match reality.
   - This detects *concept drift* — the relationship between features
     and target has changed even if feature distributions haven't.
   - Reference: Gama et al. (2014) — "A Survey on Concept Drift Adaptation"

Referência: Evidently AI — Open-source ML monitoring
            https://docs.evidentlyai.com/
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "monitoring"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Trigger 1: PSI thresholds ──────────────────────────────────────
PSI_WARNING_THRESHOLD = 0.1
PSI_RETRAIN_THRESHOLD = 0.2

# ── Trigger 2: Model staleness ─────────────────────────────────────
STALENESS_DAYS_THRESHOLD = 30  # retrain if model not updated in N days

# ── Trigger 3: Prediction error breach ─────────────────────────────
CI_CONFIDENCE_LEVEL = 0.95  # 95% confidence interval
CI_BREACH_RATIO_THRESHOLD = 0.20  # retrain if >20% of actuals fall outside CI

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two distributions.

    PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)

    Args:
        reference: Reference (training) distribution.
        current: Current (production) distribution.
        n_bins: Number of bins for histogram.

    Returns:
        PSI value. Lower is better (0 = identical distributions).
    """
    # Create bins based on reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Compute bin proportions
    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    # Normalize to proportions (add small epsilon to avoid log(0))
    eps = 1e-6
    ref_proportions = (ref_counts + eps) / (ref_counts.sum() + eps * n_bins)
    cur_proportions = (cur_counts + eps) / (cur_counts.sum() + eps * n_bins)

    # PSI formula
    psi = np.sum((cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions))

    return float(psi)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trigger 2: Model Staleness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def check_model_staleness(
    model_path: Optional[str] = None,
    max_age_days: int = STALENESS_DAYS_THRESHOLD,
) -> dict:
    """Check if the model file is older than the staleness threshold.

    A model that hasn't been retrained in a long time may silently degrade
    as market dynamics evolve (regime changes, volatility shifts, etc.).

    Args:
        model_path: Path to the model checkpoint. Defaults to best_model.pth.
        max_age_days: Maximum acceptable age in days.

    Returns:
        Dictionary with staleness check results:
            - stale (bool): True if model is older than threshold
            - age_days (float): Model age in days
            - threshold_days (int): The threshold used
            - last_modified (str): ISO timestamp of last model update
    """
    if model_path is None:
        model_path = str(ROOT_DIR / "models" / "best_model.pth")

    path = Path(model_path)
    if not path.exists():
        logger.warning("Model file not found: %s", model_path)
        return {
            "stale": True,
            "age_days": float("inf"),
            "threshold_days": max_age_days,
            "last_modified": None,
            "reason": "Model file not found",
        }

    last_modified = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - last_modified
    age_days = age.total_seconds() / 86400

    is_stale = age_days >= max_age_days

    result = {
        "stale": is_stale,
        "age_days": round(age_days, 1),
        "threshold_days": max_age_days,
        "last_modified": last_modified.isoformat(),
    }

    if is_stale:
        result["reason"] = (
            f"Model is {age_days:.0f} days old (threshold: {max_age_days} days). "
            "Financial markets evolve — periodic retraining ensures the model "
            "captures recent patterns."
        )
        logger.warning("Model staleness detected: %.0f days old (max %d)", age_days, max_age_days)
    else:
        logger.info("Model freshness OK: %.1f days old (max %d)", age_days, max_age_days)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trigger 3: Prediction Confidence Interval Breach
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def check_prediction_breach(
    predictions: np.ndarray,
    actuals: np.ndarray,
    confidence_level: float = CI_CONFIDENCE_LEVEL,
    breach_threshold: float = CI_BREACH_RATIO_THRESHOLD,
) -> dict:
    """Check if actual values are breaching the prediction confidence interval.

    Computes a confidence interval around each prediction based on the
    residual standard error, then measures the proportion of actual values
    that fall outside that interval.

    If the breach ratio exceeds the threshold, this signals **concept drift**:
    the statistical relationship between inputs and outputs has changed,
    even if the input distributions (PSI) appear stable.

    Methodology:
        1. Compute residuals: e_i = actual_i - predicted_i
        2. Estimate σ_residual = std(residuals)
        3. CI_i = predicted_i ± z_{α/2} × σ_residual
        4. breach_ratio = count(actual_i outside CI_i) / n
        5. If breach_ratio > threshold → retrain

    Args:
        predictions: Array of predicted values.
        actuals: Array of corresponding actual/observed values.
        confidence_level: Confidence level for the interval (default 0.95).
        breach_threshold: Max acceptable fraction of breaches (default 0.20).

    Returns:
        Dictionary with breach analysis:
            - breach_detected (bool): True if breach ratio exceeds threshold
            - breach_ratio (float): Fraction of actuals outside CI
            - breach_threshold (float): The threshold used
            - confidence_level (float): CI level used
            - residual_std (float): Standard deviation of residuals
            - n_breaches (int): Count of breaches
            - n_total (int): Total observations
            - mean_residual (float): Bias indicator
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    actuals = np.asarray(actuals, dtype=float).ravel()

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}"
        )

    n = len(predictions)
    if n < 5:
        logger.warning("Too few observations (%d) for reliable CI breach check.", n)
        return {
            "breach_detected": False,
            "breach_ratio": 0.0,
            "breach_threshold": breach_threshold,
            "reason": f"Insufficient data ({n} observations, need ≥ 5)",
        }

    # Step 1-2: Compute residuals and their standard deviation
    residuals = actuals - predictions
    residual_std = float(np.std(residuals, ddof=1))
    mean_residual = float(np.mean(residuals))

    # Step 3: Compute z-score for the confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * residual_std

    # Step 4: Count breaches (actuals outside CI)
    lower = predictions - margin
    upper = predictions + margin
    breaches = (actuals < lower) | (actuals > upper)
    n_breaches = int(breaches.sum())
    breach_ratio = n_breaches / n

    # Step 5: Decision
    breach_detected = breach_ratio > breach_threshold

    result = {
        "breach_detected": breach_detected,
        "breach_ratio": round(breach_ratio, 4),
        "breach_threshold": breach_threshold,
        "confidence_level": confidence_level,
        "residual_std": round(residual_std, 6),
        "mean_residual": round(mean_residual, 6),
        "ci_margin": round(margin, 6),
        "n_breaches": n_breaches,
        "n_total": n,
    }

    if breach_detected:
        result["reason"] = (
            f"{breach_ratio:.1%} of actual values fell outside the {confidence_level:.0%} "
            f"confidence interval (threshold: {breach_threshold:.0%}). "
            "This indicates concept drift — the model's learned patterns "
            "no longer match the current market behavior."
        )
        logger.warning(
            "Prediction CI breach: %.1f%% outside CI (threshold %.0f%%)",
            breach_ratio * 100,
            breach_threshold * 100,
        )
    else:
        logger.info(
            "Prediction CI OK: %.1f%% outside CI (threshold %.0f%%)",
            breach_ratio * 100,
            breach_threshold * 100,
        )

    return result


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: Optional[list[str]] = None,
    save_results: bool = True,
) -> dict:
    """Detect data drift between reference and current datasets.

    Uses PSI for each feature and optionally Evidently for detailed reports.

    Args:
        reference_data: Training/reference dataset.
        current_data: Current/production dataset.
        features: Feature columns to check. Defaults to FEATURE_COLUMNS.
        save_results: Whether to save results to JSON.

    Returns:
        Dictionary with drift detection results per feature and overall status.
    """
    features = features or FEATURE_COLUMNS
    available_features = [f for f in features if f in reference_data.columns and f in current_data.columns]

    if not available_features:
        logger.warning("No common features found between reference and current data.")
        return {"status": "error", "message": "No common features"}

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_reference": len(reference_data),
        "n_current": len(current_data),
        "features": {},
        "overall_status": "no_drift",
        "drift_detected": False,
        "retrain_recommended": False,
    }

    psi_scores = []

    for feature in available_features:
        ref_values = reference_data[feature].dropna().values.astype(float)
        cur_values = current_data[feature].dropna().values.astype(float)

        if len(ref_values) < 10 or len(cur_values) < 10:
            logger.warning("Insufficient data for feature %s. Skipping.", feature)
            continue

        psi = calculate_psi(ref_values, cur_values)
        psi_scores.append(psi)

        if psi > PSI_RETRAIN_THRESHOLD:
            status = "retrain"
            results["retrain_recommended"] = True
            results["drift_detected"] = True
        elif psi > PSI_WARNING_THRESHOLD:
            status = "warning"
            results["drift_detected"] = True
        else:
            status = "ok"

        results["features"][feature] = {
            "psi": round(psi, 6),
            "status": status,
            "ref_mean": round(float(ref_values.mean()), 4),
            "ref_std": round(float(ref_values.std()), 4),
            "cur_mean": round(float(cur_values.mean()), 4),
            "cur_std": round(float(cur_values.std()), 4),
        }

        logger.info("Feature %s: PSI=%.6f (%s)", feature, psi, status)

    # Overall PSI
    if psi_scores:
        avg_psi = sum(psi_scores) / len(psi_scores)
        results["avg_psi"] = round(avg_psi, 6)

        if results["retrain_recommended"]:
            results["overall_status"] = "retrain_recommended"
        elif results["drift_detected"]:
            results["overall_status"] = "warning"

    # Summary fields consumed by the dashboard
    results["features_analyzed"] = len(results["features"])
    results["drifted_features"] = sum(
        1 for f in results["features"].values() if f.get("status") in ("warning", "retrain")
    )
    results["method"] = "PSI"
    results["trigger_type"] = "data_drift"

    # Try Evidently for detailed report
    try:
        results["evidently_report"] = _run_evidently_report(reference_data, current_data, available_features)
    except ImportError:
        logger.info("Evidently not installed. Using PSI-only drift detection.")
    except Exception as e:
        logger.warning("Evidently report failed: %s", str(e))

    if save_results:
        output_path = RESULTS_DIR / "drift_report.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Drift report saved to %s", output_path)

    return results


def _run_evidently_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: list[str],
) -> dict:
    """Run Evidently data drift report.

    Returns:
        Summary dict from Evidently report.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data[features], current_data=current_data[features])

    report_dict = report.as_dict()

    # Extract summary
    drift_summary = {}
    metrics = report_dict.get("metrics", [])
    for metric in metrics:
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            for col, col_data in result["drift_by_columns"].items():
                drift_summary[col] = {
                    "drift_detected": col_data.get("drift_detected", False),
                    "drift_score": col_data.get("drift_score", 0),
                    "stattest_name": col_data.get("stattest_name", ""),
                }

    # Save HTML report
    try:
        html_path = RESULTS_DIR / "drift_report.html"
        report.save_html(str(html_path))
        logger.info("Evidently HTML report saved to %s", html_path)
    except Exception:
        pass

    return drift_summary


def detect_drift_from_db(
    train_ratio: float = 0.7,
    features: Optional[list[str]] = None,
    training_cutoff_date: Optional[str] = None,
) -> dict:
    """Detect drift using data from the database.

    When *training_cutoff_date* is provided the split is **date-based**:

        - Reference = data on or before the cutoff (training period).
        - Current   = data **after** the cutoff (post-training / production).

    To keep the PSI comparison meaningful (avoiding ancient-vs-modern
    distribution mismatch), only a **recent window** of the reference
    data is used — sized proportionally to the current set.

    When no cutoff is given the function falls back to a simple
    ratio-based split.

    Args:
        train_ratio: Proportion of data to use as reference (fallback).
        features: Feature columns to check.
        training_cutoff_date: ISO date string for the training-data cutoff.

    Returns:
        Drift detection results enriched with split metadata.
    """
    import sqlite3

    from src.config import DATABASE_PATH

    if not Path(DATABASE_PATH).exists():
        return {"status": "error", "message": "Database not found"}

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        df = pd.read_sql("SELECT * FROM nvidia_stock ORDER BY date", conn)
    except Exception as e:
        return {"status": "error", "message": f"Query failed: {str(e)}"}
    finally:
        conn.close()

    # Normalize column names to Title Case so they match FEATURE_COLUMNS
    df.columns = [col.capitalize() for col in df.columns]

    if len(df) < 20:
        return {"status": "error", "message": "Insufficient data"}

    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        # Strip timezone to avoid tz-naive/tz-aware comparison issues
        df["Date"] = df["Date"].dt.tz_localize(None)

    used_date_split = False
    forecast_days = 30  # model predicts up to 30 days ahead

    # ── Build reference (30 days before cutoff) vs current (30 days after) ──
    # The model forecasts up to 30 days.  Meaningful drift detection
    # compares the *last data the model trained on* against the
    # *production data it is now predicting*.  Anything older is
    # irrelevant because NVIDIA stock changed drastically over decades.

    if training_cutoff_date and "Date" in df.columns:
        cutoff = pd.Timestamp(training_cutoff_date)

        ref_start = cutoff - pd.Timedelta(days=forecast_days)
        cur_end = cutoff + pd.Timedelta(days=forecast_days)

        reference = df[(df["Date"] > ref_start) & (df["Date"] <= cutoff)]
        current = df[(df["Date"] > cutoff) & (df["Date"] <= cur_end)]

        if len(reference) >= 5 and len(current) >= 5:
            used_date_split = True
        else:
            # Not enough data in the ±30-day windows — widen progressively
            for mult in [2, 4, 8]:
                wider_ref_start = cutoff - pd.Timedelta(days=forecast_days * mult)
                wider_cur_end = cutoff + pd.Timedelta(days=forecast_days * mult)
                reference = df[(df["Date"] > wider_ref_start) & (df["Date"] <= cutoff)]
                current = df[(df["Date"] > cutoff) & (df["Date"] <= wider_cur_end)]
                if len(reference) >= 5 and len(current) >= 5:
                    used_date_split = True
                    break

        if not used_date_split:
            # Ultimate fallback: last 60 rows split 50/50
            tail = df.tail(60)
            split_idx = len(tail) // 2
            reference = tail.iloc[:split_idx]
            current = tail.iloc[split_idx:]
    else:
        # No cutoff — use last 60 trading days split 50/50
        tail = df.tail(60)
        split_idx = len(tail) // 2
        reference = tail.iloc[:split_idx]
        current = tail.iloc[split_idx:]

    result = detect_drift(reference, current, features)

    # ── Enrich with split metadata ──
    result["split_method"] = "date" if used_date_split else "ratio"
    result["analysis_window_days"] = forecast_days
    if training_cutoff_date:
        result["training_cutoff_date"] = training_cutoff_date

    if "Date" in df.columns:
        result["reference_start"] = (
            str(reference["Date"].min().date()) if len(reference) > 0 else None
        )
        result["reference_end"] = (
            str(reference["Date"].max().date()) if len(reference) > 0 else None
        )
        result["current_start"] = (
            str(current["Date"].min().date()) if len(current) > 0 else None
        )
        result["current_end"] = (
            str(current["Date"].max().date()) if len(current) > 0 else None
        )

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Combined multi-trigger detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def detect_all_triggers(
    reference_data: Optional[pd.DataFrame] = None,
    current_data: Optional[pd.DataFrame] = None,
    predictions: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    features: Optional[list[str]] = None,
    training_cutoff_date: Optional[str] = None,
    save_results: bool = True,
) -> dict:
    """Run all three retrain triggers and produce a combined report.

    Any single trigger firing is sufficient to recommend retraining.
    This implements a defense-in-depth approach to model monitoring:

        - **PSI** catches *data drift* (input distribution changes)
        - **Staleness** catches *temporal decay* (market regime evolution)
        - **CI breach** catches *concept drift* (broken input→output mapping)

    All triggers focus on **post-training data** — comparing what the
    model was trained on against what it is seeing in production.

    Args:
        reference_data: Training/reference dataset (for PSI). If None, loads from DB.
        current_data: Current/production dataset (for PSI). If None, loads from DB.
        predictions: Model predictions for CI breach check. Optional.
        actuals: Actual observed values for CI breach check. Optional.
        model_path: Path to model checkpoint for staleness check.
        features: Feature columns for PSI check.
        training_cutoff_date: ISO date string for the training-data cutoff.
        save_results: Whether to persist the combined report.

    Returns:
        Combined dictionary with all trigger results and a unified recommendation.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "triggers": {},
        "retrain_recommended": False,
        "active_triggers": [],
    }

    # ── Trigger 1: Data Drift (PSI) ────────────────────────────────
    try:
        if reference_data is not None and current_data is not None:
            psi_result = detect_drift(reference_data, current_data, features, save_results=False)
        else:
            psi_result = detect_drift_from_db(
                features=features,
                training_cutoff_date=training_cutoff_date,
            )

        report["triggers"]["data_drift"] = psi_result

        if psi_result.get("retrain_recommended", False):
            report["retrain_recommended"] = True
            report["active_triggers"].append("data_drift_psi")
    except Exception as e:
        logger.error("PSI drift check failed: %s", str(e))
        report["triggers"]["data_drift"] = {"status": "error", "message": str(e)}

    # ── Trigger 2: Model Staleness ─────────────────────────────────
    try:
        staleness_result = check_model_staleness(model_path=model_path)
        report["triggers"]["staleness"] = staleness_result

        if staleness_result.get("stale", False):
            report["retrain_recommended"] = True
            report["active_triggers"].append("model_staleness")
    except Exception as e:
        logger.error("Staleness check failed: %s", str(e))
        report["triggers"]["staleness"] = {"status": "error", "message": str(e)}

    # ── Trigger 3: Prediction CI Breach ────────────────────────────
    if predictions is not None and actuals is not None:
        try:
            breach_result = check_prediction_breach(predictions, actuals)
            report["triggers"]["prediction_breach"] = breach_result

            if breach_result.get("breach_detected", False):
                report["retrain_recommended"] = True
                report["active_triggers"].append("prediction_ci_breach")
        except Exception as e:
            logger.error("CI breach check failed: %s", str(e))
            report["triggers"]["prediction_breach"] = {"status": "error", "message": str(e)}
    else:
        report["triggers"]["prediction_breach"] = {
            "status": "skipped",
            "reason": "No predictions/actuals provided",
        }

    # ── Summary ────────────────────────────────────────────────────
    n_active = len(report["active_triggers"])
    if n_active == 0:
        report["overall_status"] = "healthy"
        report["summary"] = "All checks passed. No retraining needed."
    else:
        report["overall_status"] = "retrain_recommended"
        trigger_names = {
            "data_drift_psi": "Data Drift (PSI > 0.2)",
            "model_staleness": f"Model Staleness (≥ {STALENESS_DAYS_THRESHOLD} days)",
            "prediction_ci_breach": f"Prediction CI Breach (> {CI_BREACH_RATIO_THRESHOLD:.0%} outside CI)",
        }
        reasons = [trigger_names.get(t, t) for t in report["active_triggers"]]
        report["summary"] = (
            f"Retraining recommended — {n_active} trigger(s) active: "
            + "; ".join(reasons)
        )

    logger.info("Multi-trigger report: %s", report["summary"])

    if save_results:
        output_path = RESULTS_DIR / "multi_trigger_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Multi-trigger report saved to %s", output_path)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("MULTI-TRIGGER RETRAIN DETECTION REPORT")
    print("=" * 60)

    report = detect_all_triggers()

    print(f"\n{'Status:':<20} {report.get('overall_status', 'unknown')}")
    print(f"{'Retrain recommended:':<20} {report.get('retrain_recommended', False)}")
    print(f"{'Active triggers:':<20} {len(report.get('active_triggers', []))}")

    for trigger_name, trigger_result in report.get("triggers", {}).items():
        print(f"\n── {trigger_name} ──")
        if trigger_name == "data_drift":
            print(f"  PSI status: {trigger_result.get('overall_status', 'N/A')}")
            for feat, info in trigger_result.get("features", {}).items():
                print(f"  {feat}: PSI={info['psi']:.6f} ({info['status']})")
        elif trigger_name == "staleness":
            print(f"  Stale: {trigger_result.get('stale', 'N/A')}")
            print(f"  Age: {trigger_result.get('age_days', 'N/A')} days (max {trigger_result.get('threshold_days', 'N/A')})")
        elif trigger_name == "prediction_breach":
            status = trigger_result.get("status", "")
            if status == "skipped":
                print(f"  Skipped: {trigger_result.get('reason', 'N/A')}")
            else:
                print(f"  Breach: {trigger_result.get('breach_detected', 'N/A')}")
                print(f"  Ratio: {trigger_result.get('breach_ratio', 0):.1%} (threshold {trigger_result.get('breach_threshold', 0):.0%})")

    print(f"\n{'Summary:':<20} {report.get('summary', '')}")
    print("=" * 60)
