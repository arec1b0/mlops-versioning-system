"""
Base Model Interface
Defines abstract interface for ML models following LSP principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for ML models.
    LSP: All model implementations must follow this interface.
    """
    
    def __init__(self, **kwargs):
        """Initialize model with parameters."""
        self.model = None
        self.params = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability array
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        pass
    
    def _validate_fitted(self) -> None:
        """Validate that model is fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")