"""Tests for logs.py endpoints."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.log_database import LogEntry, LogStats


@pytest.fixture
def client():
    return TestClient(app)


def _make_log_stats(**kwargs):
    defaults = dict(
        total=100,
        by_level={"INFO": 80, "WARNING": 15, "ERROR": 5},
        by_source={"api": 60, "training": 40},
        error_rate=5.0,
        warning_rate=15.0,
        logs_per_minute=2.5,
    )
    defaults.update(kwargs)
    return LogStats(**defaults)


def _make_log_entry(id=1, message="test message"):
    return LogEntry(
        id=id,
        timestamp="2024-01-01T00:00:00",
        level="INFO",
        source="api",
        module="test_module",
        func_name="test_func",
        message=message,
        extra={},
    )


class TestLogStats:
    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_stats_returns_200(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_stats.return_value = _make_log_stats()
        mock_get.return_value = mock_db
        response = client.get("/logs/stats")
        assert response.status_code == 200

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_stats_has_expected_fields(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_stats.return_value = _make_log_stats()
        mock_get.return_value = mock_db
        response = client.get("/logs/stats")
        data = response.json()
        assert "total" in data
        assert "by_level" in data
        assert "by_source" in data
        assert "error_rate" in data
        assert "warning_rate" in data
        assert "logs_per_minute" in data
        assert "since_minutes" in data

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_stats_passes_since_param(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_stats.return_value = _make_log_stats()
        mock_get.return_value = mock_db
        client.get("/logs/stats?since=60")
        mock_db.get_stats.assert_called_once_with(since_minutes=60)

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_stats_since_minutes_in_response(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_stats.return_value = _make_log_stats()
        mock_get.return_value = mock_db
        response = client.get("/logs/stats?since=45")
        assert response.json()["since_minutes"] == 45


class TestLogTimeline:
    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_timeline_returns_200(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_timeline.return_value = [{"bucket": "2024-01-01T00:00", "count": 10}]
        mock_get.return_value = mock_db
        response = client.get("/logs/timeline")
        assert response.status_code == 200

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_timeline_has_expected_fields(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_timeline.return_value = []
        mock_get.return_value = mock_db
        response = client.get("/logs/timeline")
        data = response.json()
        assert "timeline" in data
        assert "since_minutes" in data

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_timeline_passes_since_param(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_timeline.return_value = []
        mock_get.return_value = mock_db
        client.get("/logs/timeline?since=60")
        mock_db.get_timeline.assert_called_once_with(since_minutes=60)

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_timeline_since_minutes_echoed(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_timeline.return_value = []
        mock_get.return_value = mock_db
        response = client.get("/logs/timeline?since=30")
        assert response.json()["since_minutes"] == 30


class TestLogEntries:
    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_entries_returns_200(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.query.return_value = [_make_log_entry()]
        mock_get.return_value = mock_db
        response = client.get("/logs/entries")
        assert response.status_code == 200

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_entries_has_expected_fields(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.query.return_value = [_make_log_entry(message="hello world")]
        mock_get.return_value = mock_db
        response = client.get("/logs/entries")
        data = response.json()
        assert "entries" in data
        assert "count" in data
        assert "limit" in data
        assert "offset" in data
        assert data["count"] == 1
        assert data["entries"][0]["message"] == "hello world"

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_entries_passes_filter_params(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_get.return_value = mock_db
        client.get("/logs/entries?level=ERROR&source=api&search=fail&since=60&limit=50&offset=10")
        mock_db.query.assert_called_once_with(
            level="ERROR",
            source="api",
            search="fail",
            since_minutes=60,
            limit=50,
            offset=10,
        )

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_entries_empty_result(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_get.return_value = mock_db
        response = client.get("/logs/entries")
        data = response.json()
        assert data["count"] == 0
        assert data["entries"] == []

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_entries_serializes_entry_fields(self, mock_get, client):
        mock_db = MagicMock()
        entry = LogEntry(
            id=42,
            timestamp="2024-06-01T12:00:00",
            level="WARNING",
            source="monitor",
            module="drift",
            func_name="detect",
            message="drift detected",
            extra={"psi": 0.3},
        )
        mock_db.query.return_value = [entry]
        mock_get.return_value = mock_db
        response = client.get("/logs/entries")
        e = response.json()["entries"][0]
        assert e["id"] == 42
        assert e["level"] == "WARNING"
        assert e["source"] == "monitor"
        assert e["message"] == "drift detected"


class TestLogSources:
    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_sources_returns_200(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_sources.return_value = ["api", "training", "monitor"]
        mock_get.return_value = mock_db
        response = client.get("/logs/sources")
        assert response.status_code == 200

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_sources_has_sources_key(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.get_sources.return_value = ["api", "training"]
        mock_get.return_value = mock_db
        response = client.get("/logs/sources")
        data = response.json()
        assert "sources" in data
        assert "api" in data["sources"]


class TestLogCleanup:
    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_cleanup_returns_200(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.cleanup.return_value = 50
        mock_get.return_value = mock_db
        response = client.post("/logs/cleanup")
        assert response.status_code == 200

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_cleanup_has_expected_fields(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.cleanup.return_value = 50
        mock_get.return_value = mock_db
        response = client.post("/logs/cleanup")
        data = response.json()
        assert "deleted" in data
        assert "keep_hours" in data
        assert data["deleted"] == 50

    @patch("src.utils.log_database.LogDatabase.get_instance")
    def test_cleanup_passes_keep_hours_param(self, mock_get, client):
        mock_db = MagicMock()
        mock_db.cleanup.return_value = 10
        mock_get.return_value = mock_db
        response = client.post("/logs/cleanup?keep_hours=24")
        assert response.status_code == 200
        mock_db.cleanup.assert_called_once_with(keep_hours=24)
        assert response.json()["keep_hours"] == 24


class TestFileBasedLogs:
    def test_api_log_missing_returns_fallback(self, client):
        with patch("pathlib.Path.exists", return_value=False):
            response = client.get("/logs/api")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "API log file not available." in data["content"]
        assert "lines" in data

    def test_api_log_present_returns_content(self, client):
        log_content = "2024-01-01 INFO startup\n2024-01-01 INFO running\n"
        mock_file_obj = MagicMock()
        mock_file_obj.__enter__ = MagicMock(return_value=mock_file_obj)
        mock_file_obj.__exit__ = MagicMock(return_value=False)
        mock_file_obj.readlines.return_value = log_content.splitlines(keepends=True)
        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", return_value=mock_file_obj
        ):
            response = client.get("/logs/api")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "lines" in data

    def test_training_log_no_files_returns_fallback(self, client):
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.glob", return_value=[]
        ):
            response = client.get("/logs/training")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "No training logs found." in data["content"]

    def test_services_log_returns_200(self, client):
        """Docker may or may not be available; endpoint handles both gracefully."""
        response = client.get("/logs/services")
        assert response.status_code == 200
        data = response.json()
        assert "services" in data
        assert isinstance(data["services"], list)
        assert len(data["services"]) >= 1

    def test_system_log_no_files_returns_fallback(self, client):
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.glob", return_value=[]
        ):
            response = client.get("/logs/system")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "No system logs available." in data["content"]
