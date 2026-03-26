"""
Application dependencies and shared state.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from sklearn.preprocessing import MinMaxScaler

from src.config import settings
from src.models.lstm_model import NvidiaLSTM

logger = logging.getLogger(__name__)


class ModelState:
    """Singleton to hold loaded model and scaler state."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model: Optional[NvidiaLSTM] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.model_config: dict = {}
        self.device: str = "cpu"
        self.is_training: bool = False
        self.training_run_id: Optional[str] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_loss: float = 0.0

        self._initialized = True

    def load_model(self, checkpoint_path: Optional[str] = None, scaler_path: Optional[str] = None) -> bool:
        """Load model and scaler from checkpoint."""
        try:
            import torch as _torch

            self.device = "cuda" if _torch.cuda.is_available() else "cpu"

            # Default paths
            if checkpoint_path is None:
                checkpoint_path = str(settings.model_dir / "best_model.pth")

            if scaler_path is None:
                # Try common scaler locations
                for candidate in [
                    settings.model_dir / "scaler.pkl",
                    settings.model_dir / "scaler.joblib",
                ]:
                    if candidate.exists():
                        scaler_path = str(candidate)
                        break

            # Load model
            if Path(checkpoint_path).exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                # Support both formats: full checkpoint dict or bare state_dict
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model_config = checkpoint.get("model_config", {})
                    state_dict = checkpoint["model_state_dict"]
                else:
                    # Bare state_dict saved via torch.save(model.state_dict(), ...)
                    state_dict = checkpoint
                    self.model_config = {}

                # Infer model dimensions from state_dict if config not available
                inferred_input = state_dict.get("lstm.weight_ih_l0", None)
                inferred_output = state_dict.get("fc.bias", None)
                input_size = (
                    inferred_input.shape[1] if inferred_input is not None else self.model_config.get("input_size", 5)
                )
                output_size = (
                    inferred_output.shape[0] if inferred_output is not None else self.model_config.get("output_size", 1)
                )

                self.model = NvidiaLSTM(
                    input_size=input_size,
                    hidden_size=self.model_config.get("hidden_size", settings.hidden_size),
                    num_layers=self.model_config.get("num_layers", settings.num_layers),
                    output_size=output_size,
                    dropout=self.model_config.get("dropout", settings.dropout),
                    bidirectional=self.model_config.get("bidirectional", settings.bidirectional),
                )
                self.model.load_state_dict(state_dict)
                self.model = self.model.to(self.device)
                self.model.eval()

                logger.info(f"Model loaded from {checkpoint_path}")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False

            # Load scaler
            if scaler_path and Path(scaler_path).exists():
                import pickle as _pickle

                if scaler_path.endswith(".joblib"):
                    import joblib

                    self.scaler = joblib.load(scaler_path)
                else:
                    with open(scaler_path, "rb") as f:
                        self.scaler = _pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning("Scaler not found, predictions may not work correctly")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.model is not None and self.scaler is not None


# Global model state
model_state = ModelState()


def get_model_state() -> ModelState:
    """Dependency to get model state."""
    return model_state
