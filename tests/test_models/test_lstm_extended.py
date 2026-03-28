"""Extended tests for the LSTM model module."""

import torch

from src.models.lstm_model import NvidiaLSTM, create_model


class TestNvidiaLSTMExtended:
    def test_default_params(self):
        model = NvidiaLSTM(input_size=5)
        assert model.hidden_size == 128
        assert model.num_layers == 2
        assert model.output_size == 1

    def test_custom_output_size(self):
        model = NvidiaLSTM(input_size=5, output_size=5)
        x = torch.randn(4, 10, 5)
        out = model(x)
        assert out.shape == (4, 5)

    def test_single_layer_no_dropout(self):
        """Single layer LSTM should work without dropout."""
        model = NvidiaLSTM(input_size=3, num_layers=1, dropout=0.5)
        x = torch.randn(2, 5, 3)
        out = model(x)
        assert out.shape == (2, 1)

    def test_bidirectional(self):
        model = NvidiaLSTM(input_size=3, hidden_size=16, bidirectional=True)
        x = torch.randn(2, 10, 3)
        out = model(x)
        assert out.shape == (2, 1)

    def test_predict_sequence(self):
        model = NvidiaLSTM(input_size=3, hidden_size=16, num_layers=1, output_size=3)
        device = torch.device("cpu")
        initial = torch.randn(1, 10, 3)
        preds = model.predict_sequence(initial, n_steps=5, device=device)
        assert preds.shape == (5, 3)

    def test_get_num_parameters(self):
        model = NvidiaLSTM(input_size=1, hidden_size=8, num_layers=1)
        n_params = model.get_num_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)

    def test_gradient_flow(self):
        model = NvidiaLSTM(input_size=3, hidden_size=16, num_layers=1)
        x = torch.randn(4, 10, 3, requires_grad=True)
        y = torch.randn(4, 1)
        out = model(x)
        loss = torch.nn.MSELoss()(out, y)
        loss.backward()
        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None

    def test_eval_vs_train_mode(self):
        model = NvidiaLSTM(input_size=3, hidden_size=16, num_layers=2, dropout=0.3)
        x = torch.randn(4, 10, 3)
        model.train()
        out_train = model(x).detach()
        model.eval()
        out_eval = model(x).detach()
        # Outputs may differ due to dropout
        # Just verify shapes match
        assert out_train.shape == out_eval.shape


class TestCreateModel:
    def test_factory_default(self):
        model = create_model(input_size=5)
        assert isinstance(model, NvidiaLSTM)
        assert model.input_size == 5

    def test_factory_custom(self):
        model = create_model(
            input_size=3,
            hidden_size=64,
            num_layers=3,
            dropout=0.1,
            bidirectional=True,
            output_size=3,
        )
        assert model.hidden_size == 64
        assert model.num_layers == 3
        assert model.bidirectional is True
