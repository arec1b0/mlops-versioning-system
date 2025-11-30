"""
Random Forest Model Implementation
Concrete implementation of BaseModel using Random Forest.
"""

from typing import Any, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """Random Forest Classifier implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            **kwargs: Parameters for RandomForestClassifier
        """
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(**kwargs)
        logger.info(f"Initialized RandomForestModel with params: {kwargs}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training Random Forest on {X.shape[0]} samples...")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("Training completed successfully")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class labels
        """
        self._validate_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Class probabilities
        """
        self._validate_fitted()
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()
    
    def set_params(self, **params) -> 'RandomForestModel':
        """Set model parameters."""
        self.model.set_params(**params)
        self.params.update(params)
        return self
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importances.
        
        Returns:
            Series with feature importances
        """
        self._validate_fitted()
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                name='importance'
            )
        else:
            return pd.Series()