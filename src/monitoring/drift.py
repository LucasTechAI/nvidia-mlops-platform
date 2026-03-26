"""Drift detection with Evidently.

Detects data drift by comparing current data distributions against
the training data baseline using PSI (Population Stability Index).

Thresholds:
    - PSI > 0.1 → warning (distribution shift detected)
    - PSI > 0.2 → retrain trigger (significant drift)

Referência: Evidently AI — Open-source ML monitoring
            https://docs.evidentlyai.com/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "monitoring"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PSI thresholds
PSI_WARNING_THRESHOLD = 0.1
PSI_RETRAIN_THRESHOLD = 0.2

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
) -> dict:
    """Convenience function: detect drift using data from the database.

    Splits the stored data into reference (first train_ratio%) and
    current (last 1-train_ratio%).

    Args:
        train_ratio: Proportion of data to use as reference.
        features: Feature columns to check.

    Returns:
        Drift detection results.
    """
    import sqlite3

    from src.config import DATABASE_PATH

    if not Path(DATABASE_PATH).exists():
        return {"status": "error", "message": "Database not found"}

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        df = pd.read_sql("SELECT * FROM nvidia_stock ORDER BY Date", conn)
    except Exception as e:
        return {"status": "error", "message": f"Query failed: {str(e)}"}
    finally:
        conn.close()

    if len(df) < 20:
        return {"status": "error", "message": "Insufficient data"}

    split_idx = int(len(df) * train_ratio)
    reference = df.iloc[:split_idx]
    current = df.iloc[split_idx:]

    return detect_drift(reference, current, features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = detect_drift_from_db()
    print("\n=== Drift Detection Report ===")
    print(f"Status: {results.get('overall_status', 'unknown')}")
    print(f"Retrain recommended: {results.get('retrain_recommended', False)}")
    if "features" in results:
        for feat, info in results["features"].items():
            print(f"  {feat}: PSI={info['psi']:.6f} ({info['status']})")
