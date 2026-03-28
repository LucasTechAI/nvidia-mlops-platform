"""Tests for the database manager module."""

import sqlite3

import pytest

from src.utils.database_manager import DatabaseManager, DatabaseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """DatabaseManager backed by a temporary SQLite file."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE stocks (id INTEGER PRIMARY KEY, symbol TEXT, price REAL)"
    )
    conn.execute("INSERT INTO stocks (symbol, price) VALUES ('NVDA', 100.0)")
    conn.execute("INSERT INTO stocks (symbol, price) VALUES ('AAPL', 200.0)")
    conn.commit()
    conn.close()
    return DatabaseManager(db_path)


# ---------------------------------------------------------------------------
# Tests — CRUD
# ---------------------------------------------------------------------------

class TestInsert:
    def test_insert_returns_rowid(self, db):
        row_id = db.insert(
            "INSERT INTO stocks (symbol, price) VALUES (?, ?)", ("GOOG", 150.0)
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_insert_non_insert_raises(self, db):
        with pytest.raises(DatabaseError, match="INSERT"):
            db.insert("SELECT * FROM stocks", ())

    def test_insert_bad_table_raises(self, db):
        with pytest.raises(DatabaseError):
            db.insert("INSERT INTO nonexistent (a) VALUES (?)", ("x",))


class TestInsertMany:
    def test_batch_insert(self, db):
        rows = [("TSLA", 300.0), ("AMZN", 400.0)]
        count = db.insert_many(
            "INSERT INTO stocks (symbol, price) VALUES (?, ?)", rows
        )
        assert count == 2

    def test_non_insert_raises(self, db):
        with pytest.raises(DatabaseError, match="INSERT"):
            db.insert_many("DELETE FROM stocks", [])

    def test_bad_values_type_raises(self, db):
        with pytest.raises(DatabaseError, match="list of tuples"):
            db.insert_many("INSERT INTO stocks (symbol, price) VALUES (?, ?)", "not a list")


class TestSelect:
    def test_select_all(self, db):
        rows = db.select("SELECT * FROM stocks")
        assert len(rows) == 2

    def test_select_with_params(self, db):
        rows = db.select("SELECT * FROM stocks WHERE symbol = ?", ("NVDA",))
        assert len(rows) == 1

    def test_non_select_raises(self, db):
        with pytest.raises(DatabaseError, match="SELECT"):
            db.select("DELETE FROM stocks")


class TestUpdate:
    def test_update_row(self, db):
        affected = db.update("UPDATE stocks SET price = ? WHERE symbol = ?", (999.0, "NVDA"))
        assert affected >= 1

    def test_non_update_raises(self, db):
        with pytest.raises(DatabaseError, match="UPDATE"):
            db.update("SELECT * FROM stocks", ())


class TestDelete:
    def test_delete_row(self, db):
        deleted = db.delete("DELETE FROM stocks WHERE symbol = ?", ("AAPL",))
        assert deleted == 1

    def test_non_delete_raises(self, db):
        with pytest.raises(DatabaseError, match="DELETE"):
            db.delete("SELECT * FROM stocks", ())


# ---------------------------------------------------------------------------
# Tests — table_exists / get_table_info
# ---------------------------------------------------------------------------

class TestTableUtils:
    def test_table_exists_true(self, db):
        assert db.table_exists("stocks") is True

    def test_table_exists_false(self, db):
        assert db.table_exists("nonexistent") is False

    def test_empty_table_name_raises(self, db):
        with pytest.raises(DatabaseError, match="empty"):
            db.table_exists("")

    def test_invalid_table_name_raises(self, db):
        with pytest.raises(DatabaseError, match="invalid"):
            db.table_exists("drop;table")

    def test_get_table_info_missing(self, db):
        """get_table_info for a missing table should raise."""
        with pytest.raises(DatabaseError, match="does not exist"):
            db.get_table_info("nonexistent")

    def test_get_table_info_missing_table(self, db):
        with pytest.raises(DatabaseError, match="does not exist"):
            db.get_table_info("nonexistent")
