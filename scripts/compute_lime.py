#!/usr/bin/env python3
"""
Compute LIME explainability for the trained LSTM model.
This script generates LIME explanations for test samples.
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
from src.explainability.lime_explainer import (
    explain_batch_with_lime,
    plot_lime_global,
    log_lime_to_mlflow
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
    """Compute LIME explainability."""
    logger.info("Computing LIME explainability...")
    
    # Load model
    result = load_model()
    if result is None:
        return 1
    
    model, model_config = result
    device = torch.device('cpu')
    
    # Load test data
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    X_test_path = processed_dir / 'X_test.npy'
    
    if not X_test_path.exists():
        logger.error(f"Test data not found at {X_test_path}")
        return 1
    
    X_test = np.load(X_test_path)
    logger.info(f"Loaded test data: {X_test.shape}")
    
    # Feature names
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Compute LIME explanations for a batch of samples
    # Use a subset for faster computation
    n_samples = min(50, len(X_test))
    logger.info(f"Computing LIME for {n_samples} samples...")
    
    results = explain_batch_with_lime(
        model=model,
        X=X_test,
        feature_names=feature_names,
        output_index=3,  # Explain 'Close' price (index 3)
        n_explain=n_samples,
        num_samples=300,  # Reduced for faster computation
        device=device
    )
    
    # Plot global importance
    output_dir = PROJECT_ROOT / 'outputs' / 'explainability'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plot_lime_global(
        results,
        save_path=str(output_dir / 'lime_global_importance.png')
    )
    
    logger.info(f"✅ LIME computation complete!")
    logger.info(f"   Analyzed {results['n_explained']} samples")
    logger.info(f"   Plot saved to {plot_path}")
    
    # Display feature importance
    logger.info("\n📊 LIME Global Feature Importance:")
    mean_weights_sorted = dict(sorted(results['mean_abs_weights'].items(), key=lambda x: x[1], reverse=True))
    for feature, weight in mean_weights_sorted.items():
        std = results['std_abs_weights'].get(feature, 0)
        logger.info(f"   {feature:10s}: {weight:.4f} ± {std:.4f}")
    
    # Save results as JSON
    import json
    results_path = output_dir / 'lime_explanation.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"   Results saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
