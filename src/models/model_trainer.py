"""
Model Trainer
Handles model training workflow with cross-validation.
SRP: Responsible only for training operations.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from pathlib import Path
import joblib

from src.models.base_model import BaseModel
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class ModelTrainer:
    """Trains ML models with cross-validation."""
    
    def __init__(self, model: BaseModel):
        """
        Initialize trainer with a model.
        
        Args:
            model: Model instance to train
        """
        self.model = model
        self.cv_scores: Optional[Dict[str, float]] = None
        self.paths = config.get_paths()
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> BaseModel:
        """
        Train model on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            Trained model
        """
        logger.info("Starting model training...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Train model
        self.model.train(X_train, y_train)
        
        # Validate if validation data provided
        if X_val is not None and y_val is not None:
            val_score = self.model.predict(X_val)
            logger.info(f"Validation completed on {X_val.shape[0]} samples")
        
        logger.info("Model training completed")
        
        return self.model
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(
            self.model.model,
            X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        self.cv_scores = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV {scoring}: {self.cv_scores['mean']:.4f} (+/- {self.cv_scores['std']:.4f})")
        
        return self.cv_scores
    
    def save_model(self, version: str = "v1", model_name: str = "model.joblib") -> Path:
        """
        Save trained model to disk.
        
        Args:
            version: Model version
            model_name: Name of model file
            
        Returns:
            Path to saved model
        """
        if not self.model.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        model_dir = self.paths.trained_models / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / model_name
        joblib.dump(self.model, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Model size: {model_path.stat().st_size / 1024:.2f} KB")
        
        return model_path
    
    def load_model(self, model_path: Path) -> BaseModel:
        """
        Load model from disk.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        return self.model