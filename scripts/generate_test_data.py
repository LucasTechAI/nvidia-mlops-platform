#!/usr/bin/env python3
"""
Generate test data for explainability analysis.
This script loads data, creates sequences, and saves the test split.
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
from src.data.preprocessing import (
    load_data_from_db,
    normalize_features,
    create_sequences,
    train_val_test_split
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate and save test data for explainability."""
    logger.info("Generating test data for explainability...")
    
    # Load data
    logger.info('Loading data...')
    df = load_data_from_db(
        settings.database_path,
        start_year=settings.data_start_year,
        target_column=settings.target_column
    )
    
    # Prepare features
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f'Using features: {available_features}')
    
    # Normalize (load existing scaler)
    scaler_path = settings.model_dir / 'scaler.pkl'
    if not scaler_path.exists():
        logger.error(f"Scaler not found at {scaler_path}. Please run training first.")
        return 1
    
    normalized_data, scaler = normalize_features(
        df,
        available_features,
        scaler_path=str(scaler_path)
    )
    
    # Create sequences
    X, y = create_sequences(
        normalized_data,
        sequence_length=settings.sequence_length,
        forecast_horizon=1
    )
    
    # Split data (same split as training)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y,
        train_ratio=settings.train_split,
        val_ratio=settings.val_split,
        test_ratio=settings.test_split
    )
    
    # Save test data
    processed_dir = PROJECT_ROOT / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(processed_dir / 'X_test.npy', X_test)
    np.save(processed_dir / 'y_test.npy', y_test)
    
    logger.info(f'✅ Saved test data to {processed_dir}')
    logger.info(f'   X_test shape: {X_test.shape}')
    logger.info(f'   y_test shape: {y_test.shape}')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
