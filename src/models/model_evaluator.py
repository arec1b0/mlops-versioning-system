"""
Model Evaluator
Evaluates model performance with comprehensive metrics.
SRP: Responsible only for model evaluation.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates ML model performance."""

    def __init__(self, model: BaseModel):
        """
        Initialize evaluator with a model.

        Args:
            model: Trained model instance
        """
        self.model = model
        self.metrics: Dict[str, Any] = {}

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        average: str = "binary"
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test target
            average: Averaging method for metrics

        Returns:
            Dictionary with evaluation metrics (scalar only)
        """
        logger.info("Evaluating model performance...")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Calculate scalar metrics only
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test, y_pred, average=average, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, y_pred, average=average, zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, y_pred, average=average, zero_division=0
                )
            ),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba[:, 1])),
        }

        # Confusion matrix (stored separately, not as metric)
        cm = confusion_matrix(y_test, y_pred)
        self.confusion_matrix = cm

        # Log metrics
        logger.info("=" * 50)
        logger.info("Model Performance Metrics:")
        logger.info("=" * 50)
        for metric, value in self.metrics.items():
            logger.info(f"{metric.capitalize()}: {value:.4f}")

        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info("=" * 50)

        return self.metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix array

        Raises:
            RuntimeError: If evaluate() has not been called yet.
        """
        if not hasattr(self, "confusion_matrix"):
            raise RuntimeError("Model must be evaluated first")
        return self.confusion_matrix

    def get_classification_report(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> str:
        """
        Get detailed classification report.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Classification report string
        """
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)

        logger.info("\nClassification Report:")
        logger.info("\n" + report)

        return report

    def calculate_business_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cost_fp: float = 10.0,
        cost_fn: float = 100.0,
        revenue_tp: float = 50.0
    ) -> Dict[str, float]:
        """
        Calculate business-oriented metrics.

        Args:
            X_test: Test features
            y_test: Test target
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
            revenue_tp: Revenue from true positive

        Returns:
            Dictionary with business metrics
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()

        total_cost = (fp * cost_fp) + (fn * cost_fn)
        total_revenue = tp * revenue_tp
        net_value = total_revenue - total_cost

        business_metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_cost": float(total_cost),
            "total_revenue": float(total_revenue),
            "net_value": float(net_value),
            "roi": float(
                (net_value / total_cost * 100) if total_cost > 0 else 0
            ),
        }

        logger.info("\nBusiness Metrics:")
        logger.info(f"Net Value: ${net_value:.2f}")
        logger.info(f"ROI: {business_metrics['roi']:.2f}%")
        logger.info(f"Total Cost: ${total_cost:.2f}")
        logger.info(f"Total Revenue: ${total_revenue:.2f}")

        return business_metrics

    def compare_predictions(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_size: int = 10
    ) -> pd.DataFrame:
        """
        Compare predictions vs actual values.

        Args:
            X_test: Test features
            y_test: Test target
            sample_size: Number of samples to display

        Returns:
            DataFrame with comparison
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        comparison = pd.DataFrame({
            "actual": y_test.values[:sample_size],
            "predicted": y_pred[:sample_size],
            "probability_class_0": y_pred_proba[:sample_size, 0],
            "probability_class_1": y_pred_proba[:sample_size, 1],
            "correct": (
                y_test.values[:sample_size] == y_pred[:sample_size]
            ),
        })

        logger.info(f"\nPrediction Samples (first {sample_size}):")
        logger.info(f"\n{comparison}")

        return comparison
