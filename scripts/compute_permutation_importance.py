#!/usr/bin/env python3
"""
Compute permutation importance for the trained LSTM model.
This provides a global view of feature importance.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models.lstm_model import NvidiaLSTM
from src.explainability.feature_importance import (
    compute_permutation_importance,
    plot_feature_importance,
    log_explainability_to_mlflow
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model():
    """Load the trained model from checkpoint."""
    model_path = settings.model_dir / 'best_model.pth'
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('model_config', {})
    
    model = NvidiaLSTM(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        output_size=model_config['output_size'],
        dropout=model_config.get('dropout', 0.2),
        bidirectional=model_config.get('bidirectional', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model, model_config


def main():
    """Compute permutation importance."""
    logger.info("Computing permutation importance...")
    
    # Load model
    result = load_model()
    if result is None:
        return 1
    
    model, model_config = result
    device = torch.device('cpu')
    
    # Load test data
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    X_test_path = processed_dir / 'X_test.npy'
    y_test_path = processed_dir / 'y_test.npy'
    
    if not X_test_path.exists() or not y_test_path.exists():
        logger.error(f"Test data not found at {processed_dir}")
        return 1
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    logger.info(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
    
    # Compute permutation importance
    logger.info("Computing permutation importance (this may take a moment)...")
    results = compute_permutation_importance(
        model=model,
        X=X_test,
        y=y_test,
        n_repeats=10,
        device=device
    )
    
    # Plot and save
    output_dir = PROJECT_ROOT / 'outputs' / 'explainability'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plot_feature_importance(
        results,
        save_path=str(output_dir / 'permutation_importance.png')
    )
    
    logger.info(f"✅ Permutation importance computation complete!")
    logger.info(f"   Plot saved to {plot_path}")
    
    # Display results
    logger.info("\n📊 Permutation Importance Results:")
    for i, feature in enumerate(results['feature_names']):
        mean_imp = results['importances_mean'][i]
        std_imp = results['importances_std'][i]
        logger.info(f"   {feature:10s}: {mean_imp:.6f} ± {std_imp:.6f}")
    
    logger.info(f"\n   Baseline RMSE: {results['baseline_rmse']:.6f}")
    
    # Save results as JSON
    import json
    results_path = output_dir / 'permutation_importance.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"   Results saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
