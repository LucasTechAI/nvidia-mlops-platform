"""
Evaluation Results Dashboard Component.

Displays model evaluation metrics, explainability results,
and RAGAS / LLM-judge evaluation scores.
"""

import json
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


def render_evaluation_page():
    """Render the evaluation results page."""
    st.markdown(
        '<p class="section-header">📋 Model Evaluation</p>',
        unsafe_allow_html=True,
    )

    tab_metrics, tab_explain, tab_llm = st.tabs(
        ["📊 Evaluation Metrics", "🔍 Explainability", "🤖 LLM Evaluation"]
    )

    with tab_metrics:
        _render_evaluation_metrics()

    with tab_explain:
        _render_explainability()

    with tab_llm:
        _render_llm_evaluation()


# ── Evaluation Metrics ─────────────────────────────────────────


def _render_evaluation_metrics():
    """Show model evaluation metrics from latest training."""
    st.markdown("### Model Performance Metrics")

    # Try loading champion-challenger comparison for latest metrics
    comparison_path = ROOT_DIR / "outputs" / "champion_challenger" / "latest_comparison.json"
    if comparison_path.exists():
        try:
            data = json.loads(comparison_path.read_text())
            comparison = data.get("comparison", {})
            challenger = comparison.get("challenger", {})
            if challenger:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{challenger.get('rmse', 0):.6f}")
                with col2:
                    st.metric("MAE", f"{challenger.get('mae', 0):.6f}")
                with col3:
                    st.metric("MAPE", f"{challenger.get('mape', 0):.2f}%")
                with col4:
                    st.metric("R²", f"{challenger.get('r2', 0):.4f}")
        except Exception:
            pass

    # Try loading MLflow metrics
    st.markdown("#### MLflow Experiment History")
    st.markdown(
        """
        <div class="info-box">
            <p>View detailed experiment history and compare runs in
            <a href="http://localhost:5000" target="_blank">MLflow UI</a>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show artifacts if available
    artifacts_dir = ROOT_DIR / "data" / "outputs"
    _show_training_artifacts(artifacts_dir)


def _show_training_artifacts(artifacts_dir: Path):
    """Display training artifacts (loss curves, prediction plots)."""
    if not artifacts_dir.exists():
        return

    image_files = list(artifacts_dir.glob("*.png"))
    if image_files:
        st.markdown("#### Training Artifacts")
        for img_path in sorted(image_files)[:6]:  # Limit to 6 images
            st.image(str(img_path), caption=img_path.stem.replace("_", " ").title())


# ── Explainability ─────────────────────────────────────────────


def _render_explainability():
    """Show feature importance and model explainability results."""
    st.markdown("### Feature Importance (Permutation)")
    st.markdown(
        """
        <div class="info-box">
            <p class="info-box-title">ℹ️ Permutation Importance</p>
            <p>Measures how much model performance degrades when each feature
            is randomly shuffled. Higher importance means the feature contributes
            more to predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔬 Compute Feature Importance", key="btn_explain"):
        with st.spinner("Computing permutation importance (this may take a moment)..."):
            try:
                from src.explainability.feature_importance import compute_permutation_importance

                results = compute_permutation_importance()
                st.session_state["explainability_results"] = results
            except Exception as e:
                st.error(f"Explainability computation failed: {e}")
                return

    results = st.session_state.get("explainability_results")
    if results:
        import pandas as pd

        # Display as table
        importance_data = results.get("feature_importance", {})
        if importance_data:
            df = pd.DataFrame(
                [
                    {"Feature": k, "Importance": v["mean"], "Std": v.get("std", 0)}
                    for k, v in importance_data.items()
                ]
            ).sort_values("Importance", ascending=False)
            st.dataframe(df, use_container_width=True)

            # Bar chart
            st.bar_chart(df.set_index("Feature")["Importance"])

        with st.expander("📋 Raw Results"):
            st.json(results)
    else:
        st.info("Click **Compute Feature Importance** to analyze feature contributions.")

    # Show saved explainability artifacts
    explain_dir = ROOT_DIR / "outputs" / "explainability"
    if explain_dir.exists():
        for img in sorted(explain_dir.glob("*.png"))[:3]:
            st.image(str(img), caption=img.stem.replace("_", " ").title())


# ── LLM Evaluation ────────────────────────────────────────────


def _render_llm_evaluation():
    """Show LLM-as-judge and RAGAS evaluation results."""
    st.markdown("### LLM-as-Judge Evaluation")
    st.markdown(
        """
        <div class="info-box">
            <p class="info-box-title">ℹ️ Evaluation Framework</p>
            <p>Uses LLM-as-judge for qualitative assessment and RAGAS framework
            for RAG pipeline evaluation (faithfulness, relevance, context precision).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Golden set evaluation
    golden_path = ROOT_DIR / "data" / "golden_set" / "golden_set.json"
    if golden_path.exists():
        try:
            golden_data = json.loads(golden_path.read_text())
            st.markdown(f"#### Golden Set: **{len(golden_data)}** evaluation pairs")

            with st.expander("📋 View Golden Set"):
                for i, item in enumerate(golden_data[:10], 1):
                    st.markdown(f"**Q{i}**: {item.get('query', item.get('question', ''))}")
                    expected = item.get("expected", item.get("answer", ""))
                    if expected:
                        st.markdown(f"*Expected*: {expected}")
                    st.markdown("---")
        except Exception:
            pass
    else:
        st.info("Golden set not found at `data/golden_set/golden_set.json`.")

    # Show evaluation results if available
    eval_dir = ROOT_DIR / "outputs" / "evaluation"
    if eval_dir.exists():
        result_files = list(eval_dir.glob("*.json"))
        if result_files:
            st.markdown("#### Previous Evaluation Results")
            for rf in sorted(result_files, reverse=True)[:5]:
                with st.expander(f"📄 {rf.stem}"):
                    st.json(json.loads(rf.read_text()))

    if st.button("▶️ Run LLM Evaluation", key="btn_llm_eval"):
        st.warning(
            "LLM evaluation requires API keys configured in `.env`. "
            "See `evaluation/llm_judge.py` and `evaluation/ragas_eval.py`."
        )
