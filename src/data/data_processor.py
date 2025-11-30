"""
Data Processor
Handles data preprocessing and feature engineering.
SRP: Responsible only for data transformation operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class DataProcessor:
    """Processes and transforms data for ML models."""
    
    def __init__(self):
        """Initialize data processor."""
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: Optional[list] = None
        self.target_column: str = 'target'
        self.paths = config.get_paths()
    
    def split_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'target'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Exclude metadata columns
        exclude_cols = [target_col, 'created_at', 'data_version']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_columns = feature_cols
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Target: {target_col}, shape: {y.shape}")
        
        return X, y
    
    def train_test_split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Target distribution in train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            fit: Whether to fit scaler on training data
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info("Fitted StandardScaler on training data")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Features scaled successfully")
        
        return X_train_scaled, X_test_scaled
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        version: str = "v1"
    ) -> dict:
        """
        Save processed data to disk.
        
        Args:
            X_train, X_test: Feature sets
            y_train, y_test: Target sets
            version: Data version identifier
            
        Returns:
            Dictionary with saved file paths
        """
        processed_dir = self.paths.processed_data / version
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        X_train.to_csv(processed_dir / "X_train.csv", index=False)
        X_test.to_csv(processed_dir / "X_test.csv", index=False)
        y_train.to_csv(processed_dir / "y_train.csv", index=False, header=True)
        y_test.to_csv(processed_dir / "y_test.csv", index=False, header=True)
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = processed_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
        
        paths = {
            'X_train': str(processed_dir / "X_train.csv"),
            'X_test': str(processed_dir / "X_test.csv"),
            'y_train': str(processed_dir / "y_train.csv"),
            'y_test': str(processed_dir / "y_test.csv"),
            'scaler': str(processed_dir / "scaler.joblib") if self.scaler else None
        }
        
        logger.info(f"Processed data saved to: {processed_dir}")
        
        return paths
    
    def load_scaler(self, scaler_path: Path) -> StandardScaler:
        """Load saved scaler."""
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from: {scaler_path}")
        return self.scaler