"""
Observability Dashboard Component.

Displays drift monitoring, model health, and system telemetry.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def render_observability_page():
    """Render the observability / drift monitoring page."""
    st.markdown(
        '<p class="section-header">🔍 Model Observability</p>',
        unsafe_allow_html=True,
    )

    tab_drift, tab_champion, tab_telemetry = st.tabs(["📉 Drift Detection", "🏆 Champion-Challenger", "📊 Telemetry"])

    with tab_drift:
        _render_drift_section()

    with tab_champion:
        _render_champion_challenger_section()

    with tab_telemetry:
        _render_telemetry_section()


# ── Drift Detection ───────────────────────────────────────────


def _render_drift_section():
    """Show latest drift detection results."""
    st.markdown("### Data Drift Monitoring")
    st.markdown(
        """
        <div class="info-box">
            <p class="info-box-title">ℹ️ About Drift Detection</p>
            <p>Statistical tests (KS-test, PSI) compare the distribution of recent
            predictions with training data. Drift triggers champion-challenger
            re-evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Run Drift Detection", key="btn_drift"):
        with st.spinner("Analyzing data distributions..."):
            try:
                from src.monitoring.drift import detect_drift_from_db

                results = detect_drift_from_db()
                st.session_state["drift_results"] = results
            except Exception as e:
                st.error(f"Drift detection failed: {e}")
                return

    results = st.session_state.get("drift_results")
    if results:
        drift_detected = results.get("drift_detected", False)
        if drift_detected:
            st.markdown(
                '<div class="warning-box">⚠️ <b>Drift Detected</b> — Consider retraining the model.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="success-box">✅ <b>No Drift Detected</b> — '
                "Model is operating within expected distributions.</div>",
                unsafe_allow_html=True,
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Analyzed", results.get("features_analyzed", "N/A"))
        with col2:
            st.metric(
                "Drifted Features",
                results.get("drifted_features", 0),
            )
        with col3:
            st.metric("Method", results.get("method", "KS-test"))

        with st.expander("📋 Detailed Results"):
            st.json(results)
    else:
        st.info("Click **Run Drift Detection** to analyze current data distributions.")


# ── Champion-Challenger ────────────────────────────────────────


def _render_champion_challenger_section():
    """Show champion-challenger comparison results."""
    st.markdown("### Champion-Challenger Pipeline")
    st.markdown(
        """
        <div class="info-box">
            <p class="info-box-title">ℹ️ About Champion-Challenger</p>
            <p>Automated model promotion: a challenger model is trained and compared
            against the current champion. Promotion requires ≥0.5% RMSE improvement.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load latest comparison result
    comparison_path = ROOT_DIR / "outputs" / "champion_challenger" / "latest_comparison.json"

    if comparison_path.exists():
        try:
            data = json.loads(comparison_path.read_text())
            _display_comparison(data)
        except Exception as e:
            st.warning(f"Could not load comparison data: {e}")
    else:
        st.info("No champion-challenger comparison has been run yet.")

    if st.button("🚀 Run Champion-Challenger", key="btn_cc"):
        with st.spinner("Running champion-challenger pipeline... This may take several minutes."):
            try:
                from src.training.champion_challenger import run_champion_challenger

                result = run_champion_challenger()
                st.session_state["cc_result"] = result
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")


def _display_comparison(data: dict):
    """Display formatted comparison results."""
    promoted = data.get("promoted", False)
    comparison = data.get("comparison", {})

    if promoted:
        st.markdown(
            '<div class="success-box">🏆 <b>Challenger Promoted!</b> — New model is now champion.</div>',
            unsafe_allow_html=True,
        )
    else:
        reason = data.get("reason", comparison.get("reason", ""))
        st.markdown(
            f'<div class="info-box">🛡️ <b>Champion Retained</b> — {reason}</div>',
            unsafe_allow_html=True,
        )

    if comparison:
        champion = comparison.get("champion", {})
        challenger = comparison.get("challenger", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Champion RMSE", f"{champion.get('rmse', 0):.6f}")
        with col2:
            st.metric("Challenger RMSE", f"{challenger.get('rmse', 0):.6f}")
        with col3:
            delta_pct = comparison.get("rmse_delta_pct", 0)
            st.metric(
                "RMSE Δ (%)",
                f"{delta_pct * 100:.2f}%",
                delta=f"{delta_pct * 100:.2f}%",
                delta_color="inverse",
            )

    ts = data.get("timestamp", "")
    if ts:
        st.caption(f"Last run: {ts}")

    with st.expander("📋 Full Pipeline Output"):
        st.json(data)


# ── Telemetry ──────────────────────────────────────────────────


def _render_telemetry_section():
    """Show system telemetry and monitoring links."""
    st.markdown("### System Telemetry")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📊 Grafana</h4>
                <p>Dashboards & Alerts</p>
                <a href="http://localhost:3000" target="_blank">Open Grafana →</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🔬 MLflow</h4>
                <p>Experiment Tracking</p>
                <a href="http://localhost:5000" target="_blank">Open MLflow →</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📈 Prometheus</h4>
                <p>Metrics Collection</p>
                <a href="http://localhost:9090" target="_blank">Open Prometheus →</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # API Health Check
    st.markdown("#### API Health")
    if st.button("🩺 Check API Health", key="btn_health"):
        try:
            import requests

            resp = requests.get("http://localhost:8000/health", timeout=5)
            if resp.status_code == 200:
                st.markdown(
                    '<div class="success-box">✅ API is healthy</div>',
                    unsafe_allow_html=True,
                )
                st.json(resp.json())
            else:
                st.warning(f"API returned status {resp.status_code}")
        except Exception:
            st.warning("API is not reachable at localhost:8000")

    st.caption(f"Dashboard refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
