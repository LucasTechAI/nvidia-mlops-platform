"""Tests for api/dependencies.py — ModelState class."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from src.api.dependencies import ModelState, get_model_state, model_state
from src.models.lstm_model import NvidiaLSTM


class TestModelStateSingleton:
    """ModelState is a singleton."""

    def test_singleton_returns_same_instance(self):
        a = ModelState()
        b = ModelState()
        assert a is b

    def test_global_model_state(self):
        assert isinstance(model_state, ModelState)

    def test_get_model_state_returns_global(self):
        state = get_model_state()
        assert state is model_state


class TestModelStateInit:
    """Test ModelState initial attributes."""

    def test_default_attributes(self):
        state = ModelState()
        # These should exist (may be None or False initially)
        assert hasattr(state, "model")
        assert hasattr(state, "scaler")
        assert hasattr(state, "model_config")
        assert hasattr(state, "device")
        assert hasattr(state, "is_training")
        assert hasattr(state, "current_epoch")
        assert hasattr(state, "total_epochs")
        assert hasattr(state, "current_loss")
        assert hasattr(state, "training_run_id")


class TestModelStateIsReady:
    """Test the is_ready property."""

    def test_not_ready_when_no_model(self):
        state = ModelState()
        state.model = None
        state.scaler = MagicMock()
        assert state.is_ready is False

    def test_not_ready_when_no_scaler(self):
        state = ModelState()
        state.model = MagicMock()
        state.scaler = None
        assert state.is_ready is False

    def test_ready_when_both_present(self):
        state = ModelState()
        state.model = MagicMock()
        state.scaler = MagicMock()
        assert state.is_ready is True

    def teardown_method(self):
        """Reset state."""
        state = ModelState()
        state.model = None
        state.scaler = None


class TestModelStateLoadModel:
    """Test load_model with various checkpoint formats."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset model state before/after each test."""
        state = ModelState()
        state.model = None
        state.scaler = None
        state.model_config = {}
        yield state
        state.model = None
        state.scaler = None
        state.model_config = {}

    def _create_checkpoint(self, tmpdir, include_config=True, include_scaler=True):
        """Helper: create a fake checkpoint with a real model state_dict."""
        model = NvidiaLSTM(input_size=5, hidden_size=32, num_layers=1, output_size=5)
        state_dict = model.state_dict()

        checkpoint_path = str(Path(tmpdir) / "best_model.pth")

        if include_config:
            checkpoint = {
                "model_state_dict": state_dict,
                "model_config": {
                    "input_size": 5,
                    "hidden_size": 32,
                    "num_layers": 1,
                    "output_size": 5,
                    "dropout": 0.0,
                    "bidirectional": False,
                },
            }
        else:
            # Bare state_dict
            checkpoint = state_dict

        torch.save(checkpoint, checkpoint_path)

        scaler_path = None
        if include_scaler:
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            scaler.fit(np.random.rand(10, 5))
            scaler_path = str(Path(tmpdir) / "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        return checkpoint_path, scaler_path

    def test_load_full_checkpoint_with_scaler(self, reset_state):
        state = reset_state
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path, sc_path = self._create_checkpoint(tmpdir)
            result = state.load_model(cp_path, scaler_path=sc_path)

        assert result is True
        assert state.model is not None
        assert state.scaler is not None

    def test_load_bare_state_dict(self, reset_state):
        state = reset_state
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a bare state_dict (no wrapper dict) matching default settings
            model = NvidiaLSTM(
                input_size=5,
                hidden_size=128,
                num_layers=2,
                output_size=5,
            )
            bare_path = str(Path(tmpdir) / "bare.pth")
            torch.save(model.state_dict(), bare_path)
            result = state.load_model(bare_path)

        assert result is True
        assert state.model is not None

    def test_load_missing_checkpoint_returns_false(self, reset_state):
        state = reset_state
        result = state.load_model("/nonexistent/path/model.pth")
        assert result is False

    def test_load_with_joblib_scaler(self, reset_state):
        state = reset_state
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path, _ = self._create_checkpoint(tmpdir, include_scaler=False)
            # Create joblib scaler
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            scaler.fit(np.random.rand(10, 5))
            scaler_path = str(Path(tmpdir) / "scaler.joblib")
            import joblib
            joblib.dump(scaler, scaler_path)

            result = state.load_model(cp_path, scaler_path=scaler_path)

        assert result is True
        assert state.scaler is not None

    def test_load_no_scaler_warns(self, reset_state):
        state = reset_state
        state.scaler = None  # ensure clean
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path, _ = self._create_checkpoint(tmpdir, include_scaler=False)
            # Pass an explicit non-existent scaler path to avoid default path search
            result = state.load_model(cp_path, scaler_path="/nonexistent/scaler.pkl")

        assert result is True
        # No scaler was loaded, so it should still be None
        assert state.scaler is None

    def test_load_corrupt_checkpoint_returns_false(self, reset_state):
        state = reset_state
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = str(Path(tmpdir) / "bad.pth")
            with open(bad_path, "w") as f:
                f.write("not a valid checkpoint")
            result = state.load_model(bad_path)

        assert result is False

    def test_model_in_eval_mode_after_load(self, reset_state):
        state = reset_state
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path, _ = self._create_checkpoint(tmpdir, include_scaler=False)
            state.load_model(cp_path)

        assert state.model is not None
        assert not state.model.training  # eval mode
