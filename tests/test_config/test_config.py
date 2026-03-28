"""Tests for the centralized config module."""

from pathlib import Path


class TestPathConstants:
    def test_root_dir_exists(self):
        from src.config import ROOT_DIR

        assert ROOT_DIR.exists()
        assert (ROOT_DIR / "pyproject.toml").exists()

    def test_data_dir_exists(self):
        from src.config import DATA_DIR

        assert DATA_DIR.exists()

    def test_models_dir_exists(self):
        from src.config import MODELS_DIR

        assert MODELS_DIR.exists()

    def test_logs_dir_exists(self):
        from src.config import LOGS_DIR

        assert LOGS_DIR.exists()


class TestModelConstants:
    def test_sequence_length(self):
        from src.config import SEQUENCE_LENGTH

        assert SEQUENCE_LENGTH == 60

    def test_hidden_size(self):
        from src.config import HIDDEN_SIZE

        assert HIDDEN_SIZE == 128

    def test_num_layers(self):
        from src.config import NUM_LAYERS

        assert NUM_LAYERS == 2

    def test_split_ratios_sum_to_one(self):
        from src.config import TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT

        assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-9


class TestDataConfig:
    def test_default_values(self):
        from src.config import DataConfig

        dc = DataConfig()
        assert dc.target_column == "Close"
        assert dc.scaler_type == "minmax"
        assert dc.train_split == 0.7

    def test_custom_values(self):
        from src.config import DataConfig

        dc = DataConfig(target_column="Open", train_split=0.8, val_split=0.1, test_split=0.1)
        assert dc.target_column == "Open"
        assert dc.train_split == 0.8


class TestMLflowConfig:
    def test_default_experiment_name(self):
        from src.config import MLflowConfig

        mc = MLflowConfig()
        assert mc.experiment_name == "nvidia-lstm-forecast"
        assert mc.registered_model_name == "nvidia-lstm-model"
        assert mc.log_models is True

    def test_tracking_uri_set(self):
        from src.config import MLflowConfig

        mc = MLflowConfig()
        assert mc.tracking_uri  # not empty


class TestHPOConfig:
    def test_defaults(self):
        from src.config import HPOConfig

        hpo = HPOConfig()
        assert hpo.n_trials == 50
        assert hpo.direction == "minimize"
        assert hpo.metric == "val_rmse"
        assert hpo.sampler == "TPE"
        assert 32 in hpo.hidden_size_choices
        assert 60 in hpo.sequence_length_choices


class TestPredictionConfig:
    def test_defaults(self):
        from src.config import PredictionConfig

        pc = PredictionConfig()
        assert pc.forecast_horizon == 30
        assert pc.confidence_level == 0.95
        assert pc.save_format == "csv"


class TestSettings:
    def test_singleton_exists(self):
        from src.config import settings

        assert settings is not None
        assert isinstance(settings.root_dir, Path)

    def test_from_env(self):
        from src.config import Settings

        s = Settings.from_env()
        assert s.stock_symbol == "NVDA"
        assert s.epochs == 100
        assert s.learning_rate == 0.001

    def test_nested_data_config(self):
        from src.config import settings

        assert settings.data.target_column == "Close"
