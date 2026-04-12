"""Feature Store — versioned feature management with lineage tracking.

Provides:
    - Feature set registration and versioning
    - Feature retrieval by name + version
    - Lineage tracking (source → transform → feature)
    - Point-in-time feature snapshots for training reproducibility

SQLite-backed for simplicity; easily swappable for a production store.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "feature_store.db"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS feature_sets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    description TEXT,
    schema_json TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    created_by  TEXT    DEFAULT 'system',
    num_rows    INTEGER DEFAULT 0,
    num_cols    INTEGER DEFAULT 0,
    checksum    TEXT,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS feature_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_set_id  INTEGER NOT NULL,
    timestamp       TEXT    NOT NULL,
    entity_id       TEXT    NOT NULL,
    features_json   TEXT    NOT NULL,
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

CREATE TABLE IF NOT EXISTS feature_lineage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_set_id  INTEGER NOT NULL,
    source_type     TEXT    NOT NULL,
    source_name     TEXT    NOT NULL,
    transform_name  TEXT,
    transform_params TEXT,
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

CREATE TABLE IF NOT EXISTS feature_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_set_id  INTEGER NOT NULL,
    used_by         TEXT    NOT NULL,
    used_at         TEXT    NOT NULL,
    purpose         TEXT,
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);
"""


@dataclass
class FeatureSetMeta:
    """Metadata about a registered feature set."""

    name: str
    version: int
    description: str
    schema: dict
    created_at: str
    num_rows: int
    num_cols: int
    checksum: str
    lineage: list = field(default_factory=list)


class FeatureStore:
    """SQLite-backed feature store with versioning and lineage."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "FeatureStore":
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
        conn = self._get_conn()
        conn.executescript(CREATE_TABLES)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_fsid ON feature_data(feature_set_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_ts ON feature_data(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_entity ON feature_data(entity_id);")
        conn.commit()

    @staticmethod
    def _compute_checksum(df: pd.DataFrame) -> str:
        raw = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]

    # ── Registration ──────────────────────────────────────────
    def register_feature_set(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
        source_type: str = "raw",
        source_name: str = "",
        transform_name: str = "",
        transform_params: Optional[dict] = None,
    ) -> FeatureSetMeta:
        """Register a new version of a feature set from a DataFrame."""
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        checksum = self._compute_checksum(df)

        with self._cursor() as cur:
            # Get next version
            cur.execute("SELECT MAX(version) as mv FROM feature_sets WHERE name = ?", (name,))
            row = cur.fetchone()
            version = (row["mv"] or 0) + 1

            now = datetime.utcnow().isoformat()

            cur.execute(
                """INSERT INTO feature_sets
                   (name, version, description, schema_json,
                    created_at, num_rows, num_cols, checksum)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (name, version, description, json.dumps(schema), now, len(df), len(df.columns), checksum),
            )
            fs_id = cur.lastrowid

            # Store feature data row by row
            for idx, row_data in df.iterrows():
                entity_id = str(idx)
                features = row_data.to_dict()
                # Convert numpy types to native Python
                for k, v in features.items():
                    if isinstance(v, (np.integer,)):
                        features[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        features[k] = float(v)
                    elif isinstance(v, (np.bool_,)):
                        features[k] = bool(v)
                    elif pd.isna(v):
                        features[k] = None

                cur.execute(
                    "INSERT INTO feature_data (feature_set_id, timestamp, entity_id, features_json) VALUES (?,?,?,?)",
                    (fs_id, now, entity_id, json.dumps(features)),
                )

            # Record lineage
            if source_name:
                cur.execute(
                    """INSERT INTO feature_lineage
                       (feature_set_id, source_type, source_name,
                        transform_name, transform_params, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (fs_id, source_type, source_name, transform_name, json.dumps(transform_params or {}), now),
                )

        meta = FeatureSetMeta(
            name=name,
            version=version,
            description=description,
            schema=schema,
            created_at=now,
            num_rows=len(df),
            num_cols=len(df.columns),
            checksum=checksum,
        )
        logger.info("Registered feature set '%s' v%d (%d rows, %d cols)", name, version, len(df), len(df.columns))
        return meta

    # ── Retrieval ─────────────────────────────────────────────
    def get_feature_set(self, name: str, version: Optional[int] = None) -> pd.DataFrame:
        """Retrieve a feature set. If version is None, returns the latest."""
        with self._cursor() as cur:
            if version is None:
                cur.execute(
                    "SELECT id, schema_json FROM feature_sets WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,),
                )
            else:
                cur.execute(
                    "SELECT id, schema_json FROM feature_sets WHERE name = ? AND version = ?",
                    (name, version),
                )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Feature set '{name}' v{version} not found")

            fs_id = row["id"]
            cur.execute(
                "SELECT entity_id, features_json FROM feature_data WHERE feature_set_id = ? ORDER BY entity_id",
                (fs_id,),
            )
            rows = cur.fetchall()

        records = []
        for r in rows:
            feat = json.loads(r["features_json"])
            feat["_entity_id"] = r["entity_id"]
            records.append(feat)

        df = pd.DataFrame(records)
        if "_entity_id" in df.columns:
            df = df.set_index("_entity_id")
            df.index.name = None
        return df

    def get_feature_set_meta(self, name: str, version: Optional[int] = None) -> Optional[FeatureSetMeta]:
        """Get metadata for a feature set."""
        with self._cursor() as cur:
            if version is None:
                cur.execute(
                    "SELECT * FROM feature_sets WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,),
                )
            else:
                cur.execute(
                    "SELECT * FROM feature_sets WHERE name = ? AND version = ?",
                    (name, version),
                )
            row = cur.fetchone()
            if not row:
                return None

            cur.execute(
                "SELECT * FROM feature_lineage WHERE feature_set_id = ?",
                (row["id"],),
            )
            lineage = [dict(r) for r in cur.fetchall()]

        return FeatureSetMeta(
            name=row["name"],
            version=row["version"],
            description=row["description"],
            schema=json.loads(row["schema_json"]),
            created_at=row["created_at"],
            num_rows=row["num_rows"],
            num_cols=row["num_cols"],
            checksum=row["checksum"],
            lineage=lineage,
        )

    def list_feature_sets(self) -> list[dict]:
        """List all feature sets with latest version info."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT name, MAX(version) as latest_version, COUNT(*) as total_versions,
                       MAX(created_at) as last_updated
                FROM feature_sets
                GROUP BY name
                ORDER BY name
            """)
            return [dict(r) for r in cur.fetchall()]

    def record_usage(self, name: str, version: int, used_by: str, purpose: str = ""):
        """Record when a feature set is used (for lineage tracking)."""
        with self._cursor() as cur:
            cur.execute("SELECT id FROM feature_sets WHERE name = ? AND version = ?", (name, version))
            row = cur.fetchone()
            if row:
                cur.execute(
                    "INSERT INTO feature_usage (feature_set_id, used_by, used_at, purpose) VALUES (?,?,?,?)",
                    (row["id"], used_by, datetime.utcnow().isoformat(), purpose),
                )

    def get_lineage(self, name: str, version: Optional[int] = None) -> list[dict]:
        """Get full lineage for a feature set."""
        with self._cursor() as cur:
            if version is None:
                cur.execute("SELECT id FROM feature_sets WHERE name = ? ORDER BY version DESC LIMIT 1", (name,))
            else:
                cur.execute("SELECT id FROM feature_sets WHERE name = ? AND version = ?", (name, version))
            row = cur.fetchone()
            if not row:
                return []

            cur.execute(
                """SELECT fl.*, fs.name as feature_set_name, fs.version
                   FROM feature_lineage fl
                   JOIN feature_sets fs ON fl.feature_set_id = fs.id
                   WHERE fl.feature_set_id = ?""",
                (row["id"],),
            )
            return [dict(r) for r in cur.fetchall()]

    # ── Seed demo data ────────────────────────────────────────
    def seed_sample_data(self):
        """Seed feature store with sample NVIDIA stock features."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as c FROM feature_sets")
            if cur.fetchone()["c"] > 0:
                return

        import random

        random.seed(42)

        dates = pd.date_range(end=datetime.utcnow(), periods=252, freq="B")

        # Raw price features
        raw_data = {
            "close": [100 + i * 0.5 + random.gauss(0, 3) for i in range(252)],
            "volume": [random.randint(10_000_000, 80_000_000) for _ in range(252)],
            "high": [100 + i * 0.5 + abs(random.gauss(2, 1)) for i in range(252)],
            "low": [100 + i * 0.5 - abs(random.gauss(2, 1)) for i in range(252)],
            "open": [100 + i * 0.5 + random.gauss(0, 1) for i in range(252)],
        }
        raw_df = pd.DataFrame(raw_data, index=dates)
        self.register_feature_set(
            "nvidia_raw_prices",
            raw_df,
            description="Raw NVIDIA stock OHLCV data",
            source_type="csv",
            source_name="data/raw/nvidia_stock.csv",
        )

        # Technical indicators
        close = pd.Series(raw_data["close"])
        tech_data = {
            "sma_5": close.rolling(5).mean().bfill(),
            "sma_20": close.rolling(20).mean().bfill(),
            "ema_12": close.ewm(span=12).mean(),
            "rsi_14": 50 + pd.Series([random.gauss(0, 15) for _ in range(252)]),
            "macd": close.ewm(span=12).mean() - close.ewm(span=26).mean(),
            "bollinger_upper": close.rolling(20).mean().bfill() + 2 * close.rolling(20).std().bfill(),
            "bollinger_lower": close.rolling(20).mean().bfill() - 2 * close.rolling(20).std().bfill(),
            "atr_14": pd.Series([random.uniform(1, 5) for _ in range(252)]),
            "volume_sma_20": pd.Series(raw_data["volume"]).rolling(20).mean().bfill(),
        }
        tech_df = pd.DataFrame(tech_data, index=dates)
        self.register_feature_set(
            "nvidia_technical_indicators",
            tech_df,
            description="Technical analysis indicators derived from price data",
            source_type="feature_set",
            source_name="nvidia_raw_prices",
            transform_name="technical_indicators",
            transform_params={"sma_windows": [5, 20], "ema_span": 12, "rsi_period": 14},
        )

        # Lag features (for LSTM input)
        lag_data = {}
        for lag in range(1, 6):
            lag_data[f"close_lag_{lag}"] = close.shift(lag).bfill()
            lag_data[f"return_lag_{lag}"] = close.pct_change(lag).fillna(0)
        lag_df = pd.DataFrame(lag_data, index=dates)
        self.register_feature_set(
            "nvidia_lag_features",
            lag_df,
            description="Lag features for sequence model input (5 lags of price & returns)",
            source_type="feature_set",
            source_name="nvidia_raw_prices",
            transform_name="lag_generator",
            transform_params={"n_lags": 5, "features": ["close", "return"]},
        )

        logger.info("Seeded feature store with 3 feature sets")
