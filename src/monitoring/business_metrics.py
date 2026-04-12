"""Business Metrics Tracker.

Computes and tracks financial performance metrics based on
model predictions vs actuals:
    - Cumulative P&L (Profit & Loss)
    - ROI (Return on Investment)
    - Sharpe Ratio (rolling)
    - Max Drawdown
    - Win Rate (% of correct direction predictions)
    - Prediction Accuracy Buckets

All metrics are stored in SQLite for historical tracking.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "business_metrics.db"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS business_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    actual_close    REAL    NOT NULL,
    predicted_close REAL    NOT NULL,
    prev_close      REAL,
    direction_correct INTEGER DEFAULT 0,
    absolute_error  REAL    NOT NULL,
    pct_error       REAL    NOT NULL,
    cumulative_pnl  REAL    DEFAULT 0.0,
    roi_pct         REAL    DEFAULT 0.0
);
"""

CREATE_DAILY_TABLE = """
CREATE TABLE IF NOT EXISTS daily_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    UNIQUE NOT NULL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    avg_error_pct   REAL,
    cumulative_pnl  REAL,
    total_trades    INTEGER,
    winning_trades  INTEGER,
    roi_pct         REAL,
    computed_at     TEXT    NOT NULL
);
"""


@dataclass
class BusinessSnapshot:
    """Point-in-time business performance metrics."""

    cumulative_pnl: float = 0.0
    roi_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_error_pct: float = 0.0
    total_predictions: int = 0
    winning_predictions: int = 0
    daily_returns: list = field(default_factory=list)


class BusinessMetricsTracker:
    """Track and compute business-level metrics from predictions."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "BusinessMetricsTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=10)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _init_db(self):
        with self._cursor() as cur:
            cur.execute(CREATE_TABLE)
            cur.execute(CREATE_DAILY_TABLE)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_bm_date ON business_metrics(date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ds_date ON daily_summary(date);")

    def record_prediction(
        self,
        date: str,
        actual_close: float,
        predicted_close: float,
        prev_close: Optional[float] = None,
    ):
        """Record a single prediction vs actual for P&L tracking."""
        abs_error = abs(actual_close - predicted_close)
        pct_error = (abs_error / actual_close * 100) if actual_close else 0

        # Direction correctness
        direction_correct = 0
        if prev_close is not None:
            actual_dir = actual_close - prev_close
            predicted_dir = predicted_close - prev_close
            direction_correct = 1 if (actual_dir * predicted_dir > 0) else 0

        # Simple P&L: if predicted direction correct, gain = |actual change|, else lose
        pnl_change = 0.0
        if prev_close is not None:
            actual_change = actual_close - prev_close
            if direction_correct:
                pnl_change = abs(actual_change)
            else:
                pnl_change = -abs(actual_change)

        # Get cumulative P&L
        with self._cursor() as cur:
            cur.execute("SELECT cumulative_pnl FROM business_metrics ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            prev_cum_pnl = row["cumulative_pnl"] if row else 0.0

        cum_pnl = prev_cum_pnl + pnl_change
        initial_investment = prev_close or actual_close
        roi = (cum_pnl / initial_investment * 100) if initial_investment else 0

        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO business_metrics
                   (timestamp, date, actual_close, predicted_close, prev_close,
                    direction_correct, absolute_error, pct_error, cumulative_pnl, roi_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    date,
                    actual_close,
                    predicted_close,
                    prev_close,
                    direction_correct,
                    abs_error,
                    pct_error,
                    cum_pnl,
                    roi,
                ),
            )

        logger.info(
            "Recorded prediction: date=%s actual=%.2f predicted=%.2f dir=%s pnl=%.2f",
            date, actual_close, predicted_close,
            "✅" if direction_correct else "❌", pnl_change,
        )

    def compute_snapshot(self, lookback_days: int = 30) -> BusinessSnapshot:
        """Compute current business performance snapshot."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM business_metrics ORDER BY id DESC LIMIT ?",
                (lookback_days * 5,),  # buffer
            )
            rows = cur.fetchall()

        if not rows:
            return BusinessSnapshot()

        total = len(rows)
        wins = sum(1 for r in rows if r["direction_correct"])
        errors = [r["pct_error"] for r in rows]
        cum_pnl = rows[0]["cumulative_pnl"]
        roi = rows[0]["roi_pct"]

        # Daily returns for Sharpe
        returns = []
        for i in range(len(rows) - 1):
            if rows[i + 1]["actual_close"] > 0:
                r = (rows[i]["actual_close"] - rows[i + 1]["actual_close"]) / rows[i + 1]["actual_close"]
                returns.append(r)

        sharpe = 0.0
        if returns and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Max drawdown
        cumulative = np.array([r["cumulative_pnl"] for r in reversed(rows)])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak)
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        return BusinessSnapshot(
            cumulative_pnl=round(cum_pnl, 2),
            roi_pct=round(roi, 2),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 2),
            win_rate=round(wins / total * 100, 1) if total else 0,
            avg_error_pct=round(float(np.mean(errors)), 2) if errors else 0,
            total_predictions=total,
            winning_predictions=wins,
            daily_returns=returns[-30:],
        )

    def get_pnl_history(self, limit: int = 100) -> list[dict]:
        """Get P&L time series for charting."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT date, actual_close, predicted_close, direction_correct,
                          pct_error, cumulative_pnl, roi_pct
                   FROM business_metrics ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            return [dict(r) for r in reversed(cur.fetchall())]

    def get_daily_summaries(self, limit: int = 30) -> list[dict]:
        """Get daily summary history."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in reversed(cur.fetchall())]

    def seed_sample_data(self):
        """Seed realistic sample data for demo."""
        import random
        random.seed(42)

        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM business_metrics")
            if cur.fetchone()["c"] > 0:
                return

        base_price = 170.0
        prev_close = base_price

        for i in range(60):
            day = datetime.utcnow() - timedelta(days=60 - i)
            date_str = day.strftime("%Y-%m-%d")

            # Random walk
            change = random.gauss(0.2, 3.0)
            actual = prev_close + change
            # Prediction with some noise
            pred_noise = random.gauss(0, 1.5)
            predicted = actual + pred_noise

            self.record_prediction(
                date=date_str,
                actual_close=round(actual, 2),
                predicted_close=round(predicted, 2),
                prev_close=round(prev_close, 2),
            )
            prev_close = actual

        logger.info("Seeded 60 sample business metric entries")
