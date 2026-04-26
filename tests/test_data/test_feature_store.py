"""Unit tests for FeatureStore using a real SQLite database at tmp_path."""
import pandas as pd
import pytest

from src.data.feature_store import FeatureSetMeta, FeatureStore


@pytest.fixture(autouse=True)
def reset_singleton():
    """Prevent singleton pollution between tests."""
    FeatureStore._instance = None
    yield
    FeatureStore._instance = None


@pytest.fixture
def store(tmp_path):
    return FeatureStore(db_path=tmp_path / "test_feature_store.db")


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 2000, 3000],
        }
    )


class TestFeatureStoreRegistration:
    def test_register_returns_feature_set_meta(self, store, sample_df):
        meta = store.register_feature_set("test_prices", sample_df, description="test prices")
        assert isinstance(meta, FeatureSetMeta)
        assert meta.name == "test_prices"
        assert meta.version == 1
        assert meta.num_rows == 3
        assert meta.num_cols == 3
        assert meta.description == "test prices"

    def test_register_increments_version(self, store, sample_df):
        meta1 = store.register_feature_set("test_prices", sample_df)
        meta2 = store.register_feature_set("test_prices", sample_df)
        assert meta1.version == 1
        assert meta2.version == 2

    def test_register_stores_schema(self, store, sample_df):
        meta = store.register_feature_set("test_prices", sample_df)
        assert "open" in meta.schema
        assert "close" in meta.schema
        assert "volume" in meta.schema

    def test_register_computes_checksum(self, store, sample_df):
        meta = store.register_feature_set("test_prices", sample_df)
        assert meta.checksum is not None
        assert len(meta.checksum) > 0

    def test_register_with_lineage(self, store, sample_df):
        meta = store.register_feature_set(
            "test_prices",
            sample_df,
            source_type="database",
            source_name="test.db",
            transform_name="ohlcv_loader",
            transform_params={"days": 30},
        )
        assert meta.name == "test_prices"

    def test_register_different_names_independent(self, store, sample_df):
        meta_a = store.register_feature_set("prices_a", sample_df)
        meta_b = store.register_feature_set("prices_b", sample_df)
        assert meta_a.version == 1
        assert meta_b.version == 1


class TestFeatureStoreRetrieval:
    def test_get_feature_set_returns_dataframe(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        df = store.get_feature_set("test_prices")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_get_feature_set_preserves_columns(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        df = store.get_feature_set("test_prices")
        for col in ["open", "close", "volume"]:
            assert col in df.columns

    def test_get_feature_set_retrieves_latest_when_no_version(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        df2 = sample_df.copy()
        df2["open"] = [200.0, 201.0, 202.0]
        store.register_feature_set("test_prices", df2)
        df = store.get_feature_set("test_prices")
        assert float(df["open"].iloc[0]) == pytest.approx(200.0, abs=0.01)

    def test_get_feature_set_by_version(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        df2 = sample_df.copy()
        df2["open"] = [200.0, 201.0, 202.0]
        store.register_feature_set("test_prices", df2)
        df = store.get_feature_set("test_prices", version=1)
        assert float(df["open"].iloc[0]) == pytest.approx(100.0, abs=0.01)

    def test_get_feature_set_raises_value_error_when_not_found(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.get_feature_set("nonexistent_feature_set")

    def test_get_feature_set_meta_returns_meta(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df, description="desc")
        meta = store.get_feature_set_meta("test_prices")
        assert isinstance(meta, FeatureSetMeta)
        assert meta.name == "test_prices"
        assert meta.description == "desc"
        assert meta.num_rows == 3

    def test_get_feature_set_meta_returns_none_when_not_found(self, store):
        meta = store.get_feature_set_meta("nonexistent")
        assert meta is None

    def test_get_feature_set_meta_by_version(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        store.register_feature_set("test_prices", sample_df)
        meta = store.get_feature_set_meta("test_prices", version=1)
        assert meta.version == 1

    def test_get_feature_set_meta_includes_lineage(self, store, sample_df):
        store.register_feature_set(
            "test_prices",
            sample_df,
            source_name="test.db",
            source_type="database",
        )
        meta = store.get_feature_set_meta("test_prices")
        assert isinstance(meta.lineage, list)
        assert len(meta.lineage) == 1


class TestFeatureStoreList:
    def test_list_empty_when_no_feature_sets(self, store):
        result = store.list_feature_sets()
        assert result == []

    def test_list_returns_registered_feature_sets(self, store, sample_df):
        store.register_feature_set("prices_a", sample_df)
        store.register_feature_set("prices_b", sample_df)
        result = store.list_feature_sets()
        names = [r["name"] for r in result]
        assert "prices_a" in names
        assert "prices_b" in names

    def test_list_shows_latest_version(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        store.register_feature_set("test_prices", sample_df)
        result = store.list_feature_sets()
        entry = next(r for r in result if r["name"] == "test_prices")
        assert entry["latest_version"] == 2

    def test_list_shows_total_versions(self, store, sample_df):
        for _ in range(3):
            store.register_feature_set("test_prices", sample_df)
        result = store.list_feature_sets()
        entry = next(r for r in result if r["name"] == "test_prices")
        assert entry["total_versions"] == 3


class TestFeatureStoreUsage:
    def test_record_usage_does_not_raise(self, store, sample_df):
        meta = store.register_feature_set("test_prices", sample_df)
        store.record_usage(meta.name, meta.version, used_by="test_model", purpose="training")

    def test_record_usage_on_nonexistent_does_not_raise(self, store):
        store.record_usage("nonexistent", 99, used_by="test_model")


class TestFeatureStoreLineage:
    def test_get_lineage_empty_when_no_source(self, store, sample_df):
        store.register_feature_set("test_prices", sample_df)
        lineage = store.get_lineage("test_prices")
        assert lineage == []

    def test_get_lineage_returns_lineage_entries(self, store, sample_df):
        store.register_feature_set(
            "test_prices",
            sample_df,
            source_type="database",
            source_name="nvidia.db",
            transform_name="ohlcv",
        )
        lineage = store.get_lineage("test_prices")
        assert len(lineage) == 1
        assert lineage[0]["source_name"] == "nvidia.db"
        assert lineage[0]["source_type"] == "database"

    def test_get_lineage_returns_empty_for_nonexistent(self, store):
        lineage = store.get_lineage("nonexistent")
        assert lineage == []

    def test_get_lineage_by_version(self, store, sample_df):
        store.register_feature_set(
            "test_prices", sample_df, source_name="v1.db", source_type="raw"
        )
        store.register_feature_set(
            "test_prices", sample_df, source_name="v2.db", source_type="raw"
        )
        lineage_v1 = store.get_lineage("test_prices", version=1)
        lineage_v2 = store.get_lineage("test_prices", version=2)
        assert lineage_v1[0]["source_name"] == "v1.db"
        assert lineage_v2[0]["source_name"] == "v2.db"


class TestFeatureStoreSeedData:
    def test_seed_does_not_raise(self, store):
        """seed_sample_data gracefully handles missing nvidia_stock.db."""
        store.seed_sample_data()

    def test_seed_is_idempotent_when_already_seeded(self, store, sample_df):
        """Calling seed twice with real-data lineage marker should skip on second call."""
        store.register_feature_set(
            "nvidia_raw_prices",
            sample_df,
            source_name="data/nvidia_stock.db",
            source_type="database",
        )
        store.seed_sample_data()
        result = store.list_feature_sets()
        # Should still be there, not cleared
        names = [r["name"] for r in result]
        assert "nvidia_raw_prices" in names
