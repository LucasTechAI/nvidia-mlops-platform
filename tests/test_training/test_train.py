"""Tests for the training pipeline module."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.lstm_model import NvidiaLSTM
from src.training.train import (
    evaluate_on_test,
    load_model_checkpoint,
    plot_training_history,
    save_model_checkpoint,
    set_mlflow_governance_tags,
    train_epoch,
    train_model,
    validate_epoch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_model():
    """Tiny LSTM for fast tests."""
    return NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.0, output_size=5)


@pytest.fixture
def train_data():
    """Synthetic training data: (X_train, y_train)."""
    np.random.seed(42)
    n = 64
    seq_len = 10
    feats = 5
    X = np.random.randn(n, seq_len, feats).astype(np.float32)
    y = np.random.randn(n, feats).astype(np.float32)
    return X, y


@pytest.fixture
def val_data():
    """Synthetic validation data."""
    np.random.seed(123)
    n = 32
    seq_len = 10
    feats = 5
    X = np.random.randn(n, seq_len, feats).astype(np.float32)
    y = np.random.randn(n, feats).astype(np.float32)
    return X, y


@pytest.fixture
def train_loader(train_data):
    X, y = train_data
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def val_loader(val_data):
    X, y = val_data
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)


# ---------------------------------------------------------------------------
# Tests — train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    def test_returns_float_loss(self, small_model, train_loader, device):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        loss = train_epoch(small_model, train_loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_loss_decreases_over_epochs(self, small_model, train_loader, device):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.01)
        loss_1 = train_epoch(small_model, train_loader, criterion, optimizer, device)
        for _ in range(5):
            loss_n = train_epoch(small_model, train_loader, criterion, optimizer, device)
        assert loss_n < loss_1

    def test_shape_mismatch_raises(self, device):
        """Model output_size != target size should raise ValueError."""
        model = NvidiaLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0, output_size=3)
        X = torch.randn(8, 10, 5)
        y = torch.randn(8, 5)  # mismatch: model outputs 3, target is 5
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        with pytest.raises(ValueError, match="does not match target shape"):
            train_epoch(model, loader, criterion, optimizer, device)


# ---------------------------------------------------------------------------
# Tests — validate_epoch
# ---------------------------------------------------------------------------


class TestValidateEpoch:
    def test_returns_loss_and_metrics(self, small_model, val_loader, device):
        criterion = nn.MSELoss()
        loss, metrics = validate_epoch(small_model, val_loader, criterion, device)
        assert isinstance(loss, float)
        assert loss >= 0
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mape" in metrics
        assert all(v >= 0 for v in metrics.values())

    def test_model_in_eval_mode_after(self, small_model, val_loader, device):
        criterion = nn.MSELoss()
        validate_epoch(small_model, val_loader, criterion, device)
        assert not small_model.training


# ---------------------------------------------------------------------------
# Tests — evaluate_on_test
# ---------------------------------------------------------------------------


class TestEvaluateOnTest:
    def test_returns_all_metrics(self, small_model, device):
        from sklearn.preprocessing import MinMaxScaler

        np.random.seed(42)
        n, seq, feats = 30, 10, 5
        X = np.random.randn(n, seq, feats).astype(np.float32)
        y = np.random.randn(n, feats).astype(np.float32)

        scaler = MinMaxScaler()
        scaler.fit(np.random.randn(100, feats))

        result = evaluate_on_test(small_model, (X, y), scaler, device, close_idx=3)

        expected_keys = {
            "rmse",
            "mae",
            "mape",
            "r2_score",
            "correlation",
            "directional_accuracy",
            "sharpe_ratio",
            "max_drawdown",
        }
        assert expected_keys == set(result.keys())
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Tests — plot_training_history
# ---------------------------------------------------------------------------


class TestPlotTrainingHistory:
    def test_plot_saves_file(self, tmp_path):
        train_losses = [1.0, 0.8, 0.6, 0.5]
        val_losses = [1.1, 0.9, 0.7, 0.55]
        save_path = str(tmp_path / "loss.png")
        plot_training_history(train_losses, val_losses, save_path=save_path)

        from pathlib import Path

        assert Path(save_path).exists()

    def test_plot_no_save(self):
        """Should not raise when save_path is None."""
        plot_training_history([1.0, 0.5], [1.1, 0.6], save_path=None)


# ---------------------------------------------------------------------------
# Tests — train_model
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_train_model_no_mlflow(self, small_model, train_data, val_data, device):
        config = {
            "batch_size": 16,
            "learning_rate": 0.01,
            "epochs": 3,
            "early_stopping_patience": 5,
            "optimizer": "Adam",
        }
        model, history = train_model(small_model, train_data, val_data, config, device, mlflow_tracking=False)
        assert isinstance(history, dict)
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert all(isinstance(v, float) for v in history["train_loss"])

    def test_train_model_early_stopping(self, device, train_data, val_data):
        """With patience=1 and static model, early stopping triggers."""
        model = NvidiaLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0, output_size=5)
        config = {
            "batch_size": 16,
            "learning_rate": 1e-8,  # near-zero LR → no improvement
            "epochs": 50,
            "early_stopping_patience": 2,
            "optimizer": "Adam",
        }
        _, history = train_model(model, train_data, val_data, config, device, mlflow_tracking=False)
        # Should have stopped well before 50 epochs
        assert len(history["train_loss"]) < 50

    @patch("src.training.train.mlflow")
    def test_train_model_with_mlflow(self, mock_mlflow, small_model, train_data, val_data, device):
        config = {
            "batch_size": 16,
            "learning_rate": 0.01,
            "epochs": 2,
            "early_stopping_patience": 10,
            "optimizer": "Adam",
        }
        train_model(small_model, train_data, val_data, config, device, mlflow_tracking=True)
        assert mock_mlflow.log_metric.called
        assert mock_mlflow.set_tag.called  # governance tags

    def test_sgd_optimizer(self, small_model, train_data, val_data, device):
        config = {
            "batch_size": 16,
            "learning_rate": 0.01,
            "epochs": 2,
            "early_stopping_patience": 10,
            "optimizer": "SGD",
        }
        model, history = train_model(small_model, train_data, val_data, config, device, mlflow_tracking=False)
        assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# Tests — set_mlflow_governance_tags
# ---------------------------------------------------------------------------


class TestSetMlflowGovernanceTags:
    @patch("src.training.train.mlflow")
    def test_sets_required_tags(self, mock_mlflow):
        set_mlflow_governance_tags()
        calls = mock_mlflow.set_tag.call_args_list
        tag_names = {c.args[0] for c in calls}
        expected = {
            "model_name",
            "model_version",
            "model_type",
            "owner",
            "git_sha",
            "training_data_version",
            "risk_level",
            "fairness_checked",
        }
        assert expected == tag_names

    @patch("src.training.train.subprocess.check_output", side_effect=Exception("no git"))
    @patch("src.training.train.mlflow")
    def test_git_sha_fallback(self, mock_mlflow, mock_git):
        set_mlflow_governance_tags()
        git_sha_call = [c for c in mock_mlflow.set_tag.call_args_list if c.args[0] == "git_sha"]
        assert git_sha_call[0].args[1] == "unknown"


# ---------------------------------------------------------------------------
# Tests — save / load checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_save_and_load(self, small_model, device, tmp_path):
        optimizer = torch.optim.Adam(small_model.parameters())
        path = str(tmp_path / "checkpoint.pt")

        save_model_checkpoint(small_model, optimizer, epoch=5, loss=0.42, checkpoint_path=path)

        model2 = NvidiaLSTM(input_size=5, hidden_size=16, num_layers=1, dropout=0.0, output_size=5)
        optimizer2 = torch.optim.Adam(model2.parameters())

        model2, optimizer2, epoch, loss = load_model_checkpoint(model2, optimizer2, path, device)
        assert epoch == 5
        assert abs(loss - 0.42) < 1e-6

        # Weights should match
        for p1, p2 in zip(small_model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
