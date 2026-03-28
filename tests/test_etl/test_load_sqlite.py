"""Tests for the load_sqlite_nvidia ETL module."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.etl.load_sqlite_nvidia import create_table_if_not_exists, load_csv_to_sqlite


class TestCreateTable:
    def test_creates_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        create_table_if_not_exists(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nvidia_stock'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_drops_existing_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE nvidia_stock (id INTEGER)")
        conn.execute("INSERT INTO nvidia_stock VALUES (1)")
        conn.commit()
        create_table_if_not_exists(conn)
        cursor = conn.execute("SELECT count(*) FROM nvidia_stock")
        assert cursor.fetchone()[0] == 0
        conn.close()


class TestLoadCsvToSqlite:
    def test_load_from_csv(self, tmp_path):
        # Create a test CSV
        csv_path = str(tmp_path / "test.csv")
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [95.0, 96.0],
            "Close": [102.0, 103.0],
            "Volume": [1000000, 2000000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        })
        df.to_csv(csv_path, index=False)

        db_path = str(tmp_path / "test.db")
        load_csv_to_sqlite(csv_path=csv_path, db_path=db_path)

        # Verify data was loaded
        conn = sqlite3.connect(db_path)
        result = pd.read_sql("SELECT * FROM nvidia_stock", conn)
        conn.close()

        assert len(result) == 2
        assert "close" in result.columns

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_csv_to_sqlite(
                csv_path=str(tmp_path / "nonexistent.csv"),
                db_path=str(tmp_path / "test.db"),
            )
