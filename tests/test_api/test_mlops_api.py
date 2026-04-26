"""Tests for mlops_api.py endpoints."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.feature_store import FeatureSetMeta
from src.monitoring.business_metrics import BusinessSnapshot
from src.monitoring.canary_deploy import CanaryStatus
from src.monitoring.sla_monitor import SLAReport
from src.training.model_registry import ModelVersionInfo, PromotionGateResult


@pytest.fixture
def client():
    return TestClient(app)


def _make_model_version_info():
    return ModelVersionInfo(
        model_name="test-model",
        version=1,
        stage="Production",
        source_path="models/test.pth",
        run_id="abc123",
        description="Test model",
        metrics={"rmse": 3.0},
        params={"lr": "0.001"},
        tags={"env": "prod"},
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-02T00:00:00",
    )


def _make_canary_status():
    return CanaryStatus(
        deployment_id="dep-001",
        model_name="test-model",
        canary_version=2,
        baseline_version=1,
        state="canary",
        canary_weight=10.0,
        steps_completed=1,
        total_canary_requests=100,
        total_canary_errors=2,
        current_error_rate=2.0,
        current_p95_ms=300.0,
        started_at="2024-01-01T00:00:00",
        elapsed_minutes=5.0,
    )


class TestBusinessMetricsEndpoints:
    @patch("src.monitoring.business_metrics.BusinessMetricsTracker.get_instance")
    def test_snapshot_returns_200(self, mock_get, client):
        mock_tracker = MagicMock()
        mock_tracker.compute_snapshot.return_value = BusinessSnapshot()
        mock_get.return_value = mock_tracker
        response = client.get("/business-metrics/snapshot")
        assert response.status_code == 200

    @patch("src.monitoring.business_metrics.BusinessMetricsTracker.get_instance")
    def test_snapshot_has_expected_fields(self, mock_get, client):
        snap = BusinessSnapshot(
            cumulative_pnl=1000.0,
            roi_pct=10.0,
            sharpe_ratio=1.5,
            max_drawdown=-200.0,
            win_rate=60.0,
            avg_error_pct=2.5,
            total_predictions=50,
            winning_predictions=30,
            daily_returns=[0.01, 0.02],
        )
        mock_tracker = MagicMock()
        mock_tracker.compute_snapshot.return_value = snap
        mock_get.return_value = mock_tracker
        response = client.get("/business-metrics/snapshot")
        data = response.json()
        assert data["cumulative_pnl"] == pytest.approx(1000.0)
        assert data["roi_pct"] == pytest.approx(10.0)
        assert data["win_rate"] == pytest.approx(60.0)
        assert data["total_predictions"] == 50
        assert data["daily_returns"] == [0.01, 0.02]

    @patch("src.monitoring.business_metrics.BusinessMetricsTracker.get_instance")
    def test_pnl_history_returns_200(self, mock_get, client):
        mock_tracker = MagicMock()
        mock_tracker.get_pnl_history.return_value = [{"date": "2024-01-01", "cumulative_pnl": 100.0}]
        mock_get.return_value = mock_tracker
        response = client.get("/business-metrics/pnl-history")
        assert response.status_code == 200
        assert "history" in response.json()

    @patch("src.monitoring.business_metrics.BusinessMetricsTracker.get_instance")
    def test_pnl_history_passes_days_as_limit(self, mock_get, client):
        mock_tracker = MagicMock()
        mock_tracker.get_pnl_history.return_value = []
        mock_get.return_value = mock_tracker
        client.get("/business-metrics/pnl-history?days=10")
        mock_tracker.get_pnl_history.assert_called_once_with(limit=10)

    @patch("src.monitoring.business_metrics.BusinessMetricsTracker.get_instance")
    def test_daily_summaries_returns_200(self, mock_get, client):
        mock_tracker = MagicMock()
        mock_tracker.get_daily_summaries.return_value = []
        mock_get.return_value = mock_tracker
        response = client.get("/business-metrics/daily-summaries")
        assert response.status_code == 200
        assert "summaries" in response.json()

    @patch("src.monitoring.business_metrics.BusinessMetricsTracker.get_instance")
    def test_daily_summaries_passes_days_param(self, mock_get, client):
        mock_tracker = MagicMock()
        mock_tracker.get_daily_summaries.return_value = []
        mock_get.return_value = mock_tracker
        client.get("/business-metrics/daily-summaries?days=7")
        mock_tracker.get_daily_summaries.assert_called_once_with(days=7)


class TestSLAEndpoints:
    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    def test_sla_report_returns_200(self, mock_get, client):
        mock_monitor = MagicMock()
        mock_monitor.compute_sla.return_value = SLAReport()
        mock_get.return_value = mock_monitor
        response = client.get("/sla/report")
        assert response.status_code == 200

    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    def test_sla_report_has_expected_fields(self, mock_get, client):
        mock_monitor = MagicMock()
        mock_monitor.compute_sla.return_value = SLAReport(
            uptime_pct=99.9,
            total_checks=100,
            successful_checks=99,
        )
        mock_get.return_value = mock_monitor
        response = client.get("/sla/report")
        data = response.json()
        assert "uptime_pct" in data
        assert "total_checks" in data
        assert "overall_sla_met" in data
        assert data["uptime_pct"] == pytest.approx(99.9)

    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    def test_sla_report_passes_period_param(self, mock_get, client):
        mock_monitor = MagicMock()
        mock_monitor.compute_sla.return_value = SLAReport()
        mock_get.return_value = mock_monitor
        client.get("/sla/report?period_minutes=120")
        mock_monitor.compute_sla.assert_called_once_with(period_minutes=120)

    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    def test_uptime_history_returns_200(self, mock_get, client):
        mock_monitor = MagicMock()
        mock_monitor.get_uptime_history.return_value = [{"date": "2024-01-01", "uptime": 99.9}]
        mock_get.return_value = mock_monitor
        response = client.get("/sla/uptime-history")
        assert response.status_code == 200
        assert "history" in response.json()

    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    def test_uptime_history_passes_days_param(self, mock_get, client):
        mock_monitor = MagicMock()
        mock_monitor.get_uptime_history.return_value = []
        mock_get.return_value = mock_monitor
        client.get("/sla/uptime-history?days=14")
        mock_monitor.get_uptime_history.assert_called_once_with(days=14)


class TestFeatureStoreEndpoints:
    @patch("src.data.feature_store.FeatureStore.get_instance")
    def test_list_feature_sets_returns_200(self, mock_get, client):
        mock_store = MagicMock()
        mock_store.list_feature_sets.return_value = [{"name": "test_features", "latest_version": 1}]
        mock_get.return_value = mock_store
        response = client.get("/feature-store/list")
        assert response.status_code == 200
        assert "feature_sets" in response.json()

    @patch("src.data.feature_store.FeatureStore.get_instance")
    def test_get_feature_set_info_returns_200(self, mock_get, client):
        mock_store = MagicMock()
        mock_store.get_feature_set_meta.return_value = FeatureSetMeta(
            name="test_features",
            version=1,
            description="test",
            schema={"col": "float64"},
            created_at="2024-01-01T00:00:00",
            num_rows=100,
            num_cols=1,
            checksum="abc123",
        )
        mock_get.return_value = mock_store
        response = client.get("/feature-store/test_features")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_features"
        assert data["version"] == 1

    @patch("src.data.feature_store.FeatureStore.get_instance")
    def test_get_feature_set_info_returns_404_when_not_found(self, mock_get, client):
        mock_store = MagicMock()
        mock_store.get_feature_set_meta.return_value = None
        mock_get.return_value = mock_store
        response = client.get("/feature-store/nonexistent")
        assert response.status_code == 404

    @patch("src.data.feature_store.FeatureStore.get_instance")
    def test_get_feature_lineage_returns_200(self, mock_get, client):
        mock_store = MagicMock()
        mock_store.get_lineage.return_value = [{"source_type": "database", "source_name": "test.db"}]
        mock_get.return_value = mock_store
        response = client.get("/feature-store/test_features/lineage")
        assert response.status_code == 200
        data = response.json()
        assert "lineage" in data
        assert len(data["lineage"]) == 1

    @patch("src.data.feature_store.FeatureStore.get_instance")
    def test_preview_feature_set_returns_200(self, mock_get, client):
        mock_store = MagicMock()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        mock_store.get_feature_set.return_value = df
        mock_get.return_value = mock_store
        response = client.get("/feature-store/test_features/preview")
        assert response.status_code == 200
        data = response.json()
        assert "columns" in data
        assert "data" in data
        assert "total_rows" in data
        assert data["total_rows"] == 3

    @patch("src.data.feature_store.FeatureStore.get_instance")
    def test_preview_feature_set_returns_404_on_value_error(self, mock_get, client):
        mock_store = MagicMock()
        mock_store.get_feature_set.side_effect = ValueError("Feature set not found")
        mock_get.return_value = mock_store
        response = client.get("/feature-store/nonexistent/preview")
        assert response.status_code == 404


class TestModelRegistryEndpoints:
    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_list_models_returns_200(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.list_models.return_value = [{"name": "test-model"}]
        mock_get.return_value = mock_registry
        response = client.get("/model-registry/models")
        assert response.status_code == 200
        assert "models" in response.json()

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_list_model_versions_returns_200(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.list_versions.return_value = [{"version": 1, "stage": "Production"}]
        mock_get.return_value = mock_registry
        response = client.get("/model-registry/test-model/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert "versions" in data

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_get_production_model_returns_200(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.get_production_version.return_value = _make_model_version_info()
        mock_get.return_value = mock_registry
        response = client.get("/model-registry/test-model/production")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["stage"] == "Production"

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_get_production_model_returns_404_when_none(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.get_production_version.return_value = None
        mock_get.return_value = mock_registry
        response = client.get("/model-registry/test-model/production")
        assert response.status_code == 404

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_promote_model_returns_200_when_gate_passes(self, mock_get, client):
        mock_registry = MagicMock()
        gate = PromotionGateResult(passed=True, version=1, current_stage="Staging", target_stage="Production")
        mock_registry.check_promotion_gate.return_value = gate
        mock_registry.transition_stage.return_value = None
        mock_get.return_value = mock_registry
        response = client.post("/model-registry/test-model/promote/1?target_stage=Production")
        assert response.status_code == 200
        assert response.json()["promoted"] is True

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_promote_model_returns_not_promoted_when_gate_fails(self, mock_get, client):
        mock_registry = MagicMock()
        gate = PromotionGateResult(
            passed=False,
            version=1,
            current_stage="None",
            target_stage="Production",
            reason="RMSE too high",
        )
        mock_registry.check_promotion_gate.return_value = gate
        mock_get.return_value = mock_registry
        response = client.post("/model-registry/test-model/promote/1?target_stage=Production")
        assert response.status_code == 200
        assert response.json()["promoted"] is False

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_rollback_model_returns_200(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.rollback_production.return_value = _make_model_version_info()
        mock_get.return_value = mock_registry
        response = client.post("/model-registry/test-model/rollback")
        assert response.status_code == 200
        data = response.json()
        assert data["rolled_back"] is True
        assert "restored_version" in data

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_rollback_model_returns_404_when_none(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.rollback_production.return_value = None
        mock_get.return_value = mock_registry
        response = client.post("/model-registry/test-model/rollback")
        assert response.status_code == 404

    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_model_history_returns_200(self, mock_get, client):
        mock_registry = MagicMock()
        mock_registry.get_transition_history.return_value = [{"from_stage": "Staging", "to_stage": "Production"}]
        mock_get.return_value = mock_registry
        response = client.get("/model-registry/test-model/history")
        assert response.status_code == 200
        assert "history" in response.json()


class TestCanaryEndpoints:
    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_list_deployments_returns_200(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.list_deployments.return_value = []
        mock_get.return_value = mock_mgr
        response = client.get("/canary/deployments")
        assert response.status_code == 200
        assert "deployments" in response.json()

    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_list_deployments_passes_model_name(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.list_deployments.return_value = []
        mock_get.return_value = mock_mgr
        client.get("/canary/deployments?model_name=test-model")
        mock_mgr.list_deployments.assert_called_once_with("test-model", 20)

    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_canary_status_returns_200(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.get_status.return_value = _make_canary_status()
        mock_get.return_value = mock_mgr
        response = client.get("/canary/dep-001/status")
        assert response.status_code == 200
        data = response.json()
        assert data["deployment_id"] == "dep-001"
        assert data["state"] == "canary"

    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_canary_status_returns_404_when_not_found(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.get_status.return_value = None
        mock_get.return_value = mock_mgr
        response = client.get("/canary/dep-999/status")
        assert response.status_code == 404

    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_start_canary_returns_200(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.start_canary.return_value = "dep-001"
        mock_get.return_value = mock_mgr
        response = client.post("/canary/start?model_name=test-model&canary_version=2&baseline_version=1")
        assert response.status_code == 200
        data = response.json()
        assert data["deployment_id"] == "dep-001"
        assert data["state"] == "canary"
        assert data["canary_weight"] == 5

    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_evaluate_canary_returns_200(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.evaluate_step.return_value = "continue"
        mock_mgr.get_status.return_value = _make_canary_status()
        mock_get.return_value = mock_mgr
        response = client.post("/canary/dep-001/evaluate")
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "continue"
        assert data["status"] is not None

    @patch("src.monitoring.canary_deploy.CanaryDeployManager.get_instance")
    def test_canary_rollback_history_returns_200(self, mock_get, client):
        mock_mgr = MagicMock()
        mock_mgr.get_rollback_history.return_value = [{"deployment_id": "dep-001", "reason": "error rate exceeded"}]
        mock_get.return_value = mock_mgr
        response = client.get("/canary/rollback-history")
        assert response.status_code == 200
        data = response.json()
        assert "rollbacks" in data
        assert len(data["rollbacks"]) == 1


class TestCostAnalysis:
    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_cost_analysis_returns_200(self, mock_reg, mock_sla, client):
        mock_registry = MagicMock()
        mock_registry.list_versions.return_value = [{"version": 1}]
        mock_reg.return_value = mock_registry
        mock_monitor = MagicMock()
        mock_monitor.get_report.return_value = {"total_requests": 1000}
        mock_sla.return_value = mock_monitor
        response = client.get("/cost-analysis")
        assert response.status_code == 200

    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_cost_analysis_has_expected_structure(self, mock_reg, mock_sla, client):
        mock_registry = MagicMock()
        mock_registry.list_versions.return_value = []
        mock_reg.return_value = mock_registry
        mock_monitor = MagicMock()
        mock_monitor.get_report.return_value = {"total_requests": 500}
        mock_sla.return_value = mock_monitor
        response = client.get("/cost-analysis")
        data = response.json()
        assert "grand_total" in data
        assert "infra_breakdown" in data
        assert "llm_breakdown" in data
        assert "period_days" in data
        assert "infra_total" in data
        assert "llm_total" in data
        assert isinstance(data["grand_total"], (int, float))
        assert isinstance(data["infra_breakdown"], list)

    @patch("src.monitoring.sla_monitor.SLAMonitor.get_instance")
    @patch("src.training.model_registry.ModelRegistry.get_instance")
    def test_cost_analysis_respects_days_param(self, mock_reg, mock_sla, client):
        mock_registry = MagicMock()
        mock_registry.list_versions.return_value = []
        mock_reg.return_value = mock_registry
        mock_monitor = MagicMock()
        mock_monitor.get_report.return_value = {"total_requests": 0}
        mock_sla.return_value = mock_monitor
        response = client.get("/cost-analysis?days=7")
        assert response.status_code == 200
        assert response.json()["period_days"] == 7
