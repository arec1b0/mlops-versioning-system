"""
Data Loader
Handles loading data from various sources.
SRP: Responsible only for data loading operations.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import json

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class DataLoader:
    """Loads data from files with validation."""
    
    def __init__(self):
        """Initialize data loader."""
        self.paths = config.get_paths()
    
    def load_csv(self, file_path: Path, validate: bool = True) -> pd.DataFrame:
        """
        Load CSV file with validation.
        
        Args:
            file_path: Path to CSV file
            validate: Whether to validate data
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        if validate:
            self._validate_dataframe(df)
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame integrity.
        
        Args:
            df: DataFrame to validate
        """
        # Check for empty dataframe
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        if missing_pct.any():
            logger.warning(f"Missing values detected:\n{missing_pct[missing_pct > 0]}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        logger.info("Data validation completed")
    
    def load_train_test_split(
        self,
        train_path: Path,
        test_path: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test datasets.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = self.load_csv(train_path)
        test_df = self.load_csv(test_path)
        
        logger.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")
        
        return train_df, test_df
    
    def get_latest_data_version(self, data_dir: Path) -> Optional[Path]:
        """
        Get the latest version of data file.
        
        Args:
            data_dir: Directory containing data files
            
        Returns:
            Path to latest data file or None
        """
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return None
        
        # Sort by modification time
        latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Latest data file: {latest_file.name}")
        
        return latest_file