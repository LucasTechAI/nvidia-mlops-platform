"""Unit tests for BusinessMetricsTracker using a real SQLite in-memory path."""

import pytest

from src.monitoring.business_metrics import BusinessMetricsTracker, BusinessSnapshot


@pytest.fixture(autouse=True)
def reset_singleton():
    """Prevent singleton pollution between tests."""
    BusinessMetricsTracker._instance = None
    yield
    BusinessMetricsTracker._instance = None


@pytest.fixture
def tracker(tmp_path):
    return BusinessMetricsTracker(db_path=tmp_path / "test.db")


class TestRecordPrediction:
    def test_record_single_prediction(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=100.0,
            predicted_close=102.0,
        )
        history = tracker.get_pnl_history(limit=10)
        assert len(history) == 1
        assert history[0]["date"] == "2024-01-01"
        assert history[0]["actual_close"] == pytest.approx(100.0)
        assert history[0]["predicted_close"] == pytest.approx(102.0)

    def test_record_multiple_predictions(self, tracker):
        for i in range(5):
            tracker.record_prediction(
                date=f"2024-01-0{i + 1}",
                actual_close=100.0 + i,
                predicted_close=101.0 + i,
            )
        history = tracker.get_pnl_history(limit=10)
        assert len(history) == 5

    def test_direction_correct_when_both_up(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=105.0,
            predicted_close=103.0,
            prev_close=100.0,
        )
        history = tracker.get_pnl_history(limit=1)
        assert history[0]["direction_correct"] == 1

    def test_direction_incorrect_when_opposing(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=95.0,
            predicted_close=105.0,
            prev_close=100.0,
        )
        history = tracker.get_pnl_history(limit=1)
        assert history[0]["direction_correct"] == 0

    def test_pct_error_computed_correctly(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=100.0,
            predicted_close=110.0,
        )
        history = tracker.get_pnl_history(limit=1)
        assert history[0]["pct_error"] == pytest.approx(10.0, abs=0.01)

    def test_cumulative_pnl_accumulates(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=105.0,
            predicted_close=103.0,
            prev_close=100.0,
        )
        tracker.record_prediction(
            date="2024-01-02",
            actual_close=110.0,
            predicted_close=108.0,
            prev_close=105.0,
        )
        history = tracker.get_pnl_history(limit=10)
        assert history[-1]["cumulative_pnl"] > 0


class TestComputeSnapshot:
    def test_empty_db_returns_default_snapshot(self, tracker):
        snap = tracker.compute_snapshot()
        assert isinstance(snap, BusinessSnapshot)
        assert snap.total_predictions == 0
        assert snap.cumulative_pnl == 0.0
        assert snap.win_rate == 0

    def test_snapshot_with_data(self, tracker):
        for i in range(10):
            tracker.record_prediction(
                date=f"2024-01-{i + 1:02d}",
                actual_close=100.0 + i,
                predicted_close=101.0 + i,
                prev_close=99.0 + i,
            )
        snap = tracker.compute_snapshot()
        assert snap.total_predictions == 10
        assert snap.winning_predictions >= 0
        assert 0.0 <= snap.win_rate <= 100.0

    def test_snapshot_roi_and_pnl(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=105.0,
            predicted_close=103.0,
            prev_close=100.0,
        )
        snap = tracker.compute_snapshot()
        assert snap.cumulative_pnl != 0.0

    def test_snapshot_avg_error_pct(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=100.0,
            predicted_close=110.0,
        )
        snap = tracker.compute_snapshot()
        assert snap.avg_error_pct == pytest.approx(10.0, abs=0.1)

    def test_snapshot_daily_returns_limited_to_30(self, tracker):
        for i in range(40):
            tracker.record_prediction(
                date=f"2024-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}",
                actual_close=100.0 + i,
                predicted_close=101.0 + i,
                prev_close=99.0 + i,
            )
        snap = tracker.compute_snapshot()
        assert len(snap.daily_returns) <= 30


class TestGetPnlHistory:
    def test_pnl_history_empty(self, tracker):
        assert tracker.get_pnl_history() == []

    def test_pnl_history_respects_limit(self, tracker):
        for i in range(20):
            tracker.record_prediction(
                date=f"2024-01-{i + 1:02d}",
                actual_close=100.0 + i,
                predicted_close=101.0 + i,
            )
        history = tracker.get_pnl_history(limit=5)
        assert len(history) == 5

    def test_pnl_history_has_expected_keys(self, tracker):
        tracker.record_prediction(
            date="2024-01-01",
            actual_close=100.0,
            predicted_close=102.0,
        )
        history = tracker.get_pnl_history()
        row = history[0]
        assert "date" in row
        assert "actual_close" in row
        assert "predicted_close" in row
        assert "cumulative_pnl" in row
        assert "roi_pct" in row

    def test_pnl_history_chronological_order(self, tracker):
        for i in range(3):
            tracker.record_prediction(
                date=f"2024-01-0{i + 1}",
                actual_close=100.0 + i,
                predicted_close=101.0 + i,
            )
        history = tracker.get_pnl_history()
        dates = [r["date"] for r in history]
        assert dates == sorted(dates)


class TestGetDailySummaries:
    def test_daily_summaries_empty(self, tracker):
        assert tracker.get_daily_summaries() == []

    def test_daily_summaries_uses_limit_param(self, tracker):
        result = tracker.get_daily_summaries(limit=10)
        assert isinstance(result, list)


class TestSeedSampleData:
    def test_seed_creates_60_entries(self, tracker):
        tracker.seed_sample_data()
        history = tracker.get_pnl_history(limit=200)
        assert len(history) == 60

    def test_seed_is_idempotent(self, tracker):
        tracker.seed_sample_data()
        tracker.seed_sample_data()
        history = tracker.get_pnl_history(limit=200)
        assert len(history) == 60

    def test_snapshot_after_seed(self, tracker):
        tracker.seed_sample_data()
        snap = tracker.compute_snapshot()
        assert snap.total_predictions > 0


class TestPopulateFromBacktest:
    def test_returns_false_when_model_not_ready(self, tracker):
        """Empty DB, model not ready → returns False (model check fails)."""

        class FakeModelState:
            is_ready = False

        result = tracker.populate_from_backtest(FakeModelState())
        assert result is False

    def test_returns_false_when_data_is_recent(self, tracker):
        """DB with recent data → staleness check fires → returns False."""
        tracker.seed_sample_data()

        class FakeModelState:
            is_ready = True

        result = tracker.populate_from_backtest(FakeModelState())
        assert result is False

    def test_returns_false_when_exception_during_backtest(self, tracker):
        """Exception during backtest processing → returns False."""

        class BrokenModelState:
            is_ready = True
            model_config = {"sequence_length": 60}

            @property
            def model(self):
                raise RuntimeError("model broken")

            @property
            def scaler(self):
                raise RuntimeError("scaler broken")

        result = tracker.populate_from_backtest(BrokenModelState())
        assert result is False
