"""Tests for data/preprocessing load_data_from_db and related functions."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import load_data_from_db


@pytest.fixture
def stock_db(tmp_path):
    """Create a test SQLite database with nvidia_stock table."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE nvidia_stock (
            Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER
        )
    """)
    dates = pd.date_range("2018-01-01", periods=100, freq="D")
    for d in dates:
        conn.execute(
            "INSERT INTO nvidia_stock VALUES (?, ?, ?, ?, ?, ?)",
            (d.isoformat(), 100.0, 110.0, 90.0, 105.0, 1000000),
        )
    conn.commit()
    conn.close()
    return db_path


class TestLoadDataFromDb:
    def test_loads_data(self, stock_db):
        df = load_data_from_db(stock_db, start_year=2018)
        assert len(df) == 100
        assert "Close" in df.columns

    def test_filters_by_year(self, stock_db):
        # All data is from 2018, so filtering to 2020 should raise ValueError
        with pytest.raises(ValueError, match="No data found"):
            load_data_from_db(stock_db, start_year=2020)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data_from_db("/nonexistent/path.db")

    def test_missing_target_column(self, tmp_path):
        db_path = str(tmp_path / "bad.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE nvidia_stock (Date TEXT, Foo REAL)")
        conn.execute("INSERT INTO nvidia_stock VALUES ('2020-01-01', 1.0)")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError, match="Target column"):
            load_data_from_db(db_path, target_column="Close")

    def test_lowercase_columns(self, tmp_path):
        """Database with lowercase columns should be capitalized."""
        db_path = str(tmp_path / "lower.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE nvidia_stock (date TEXT, close REAL, open REAL)")
        conn.execute("INSERT INTO nvidia_stock VALUES ('2020-01-01', 100, 90)")
        conn.commit()
        conn.close()

        df = load_data_from_db(db_path, start_year=2020, target_column="Close")
        assert "Close" in df.columns
