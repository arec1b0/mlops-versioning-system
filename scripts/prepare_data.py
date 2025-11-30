"""
Data Preparation Pipeline
End-to-end pipeline for generating, processing, and versioning data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger
from src.data.data_generator import generate_and_save_data
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.versioning.dvc_manager import DVCManager

logger = get_logger(__name__)


def main():
    """Run complete data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Initialize components
    paths = config.get_paths()
    loader = DataLoader()
    processor = DataProcessor()
    dvc = DVCManager()
    
    # Step 1: Generate raw data
    logger.info("\n[Step 1/5] Generating raw data...")
    raw_data_path = paths.raw_data / "customer_data.csv"
    df = generate_and_save_data(raw_data_path, n_samples=5000, n_features=20)
    
    # Step 2: Add raw data to DVC
    logger.info("\n[Step 2/5] Adding raw data to DVC...")
    dvc.add(raw_data_path)
    
    # Step 3: Load and validate data
    logger.info("\n[Step 3/5] Loading and validating data...")
    df = loader.load_csv(raw_data_path)
    
    # Step 4: Process data
    logger.info("\n[Step 4/5] Processing data...")
    X, y = processor.split_features_target(df)
    X_train, X_test, y_train, y_test = processor.train_test_split_data(X, y)
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # Step 5: Save processed data
    logger.info("\n[Step 5/5] Saving processed data...")
    saved_paths = processor.save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, version="v1"
    )
    
    # Add processed data to DVC
    for key, path in saved_paths.items():
        if path:
            dvc.add(Path(path))
    
    logger.info("\n" + "=" * 60)
    logger.info("Data Preparation Pipeline Completed Successfully!")
    logger.info("=" * 60)
    logger.info(f"\nRaw data: {raw_data_path}")
    logger.info(f"Processed data: {paths.processed_data / 'v1'}")
    logger.info(f"\nNext steps:")
    logger.info("  1. Commit .dvc files to git")
    logger.info("  2. Push data to DVC remote: dvc push")
    logger.info("  3. Train model using processed data")


if __name__ == "__main__":
    main()