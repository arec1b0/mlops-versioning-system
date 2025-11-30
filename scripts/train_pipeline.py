"""
Model Training Pipeline
End-to-end pipeline for training and tracking models.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger
from src.data.data_loader import DataLoader
from src.models.random_forest_model import RandomForestModel
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.versioning.mlflow_tracker import MLflowTracker
from src.versioning.dvc_manager import DVCManager

logger = get_logger(__name__)


def main():
    """Run complete model training pipeline."""
    logger.info("=" * 70)
    logger.info("Starting Model Training Pipeline")
    logger.info("=" * 70)
    
    # Initialize components
    paths = config.get_paths()
    loader = DataLoader()
    mlflow_tracker = MLflowTracker()
    dvc = DVCManager()
    
    # Model hyperparameters
    model_params = {
        'n_estimators': config.get('training.hyperparameters.n_estimators', 100),
        'max_depth': config.get('training.hyperparameters.max_depth', 10),
        'min_samples_split': config.get('training.hyperparameters.min_samples_split', 2),
        'min_samples_leaf': config.get('training.hyperparameters.min_samples_leaf', 1),
        'random_state': config.get('model.random_state', 42),
        'n_jobs': -1,
        'verbose': 1
    }
    
    try:
        # Start MLflow run
        run_name = f"rf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_tracker.start_run(
            run_name=run_name,
            tags={'stage': 'training', 'model_type': 'random_forest'}
        )
        
        # Step 1: Load processed data
        logger.info("\n[Step 1/6] Loading processed data...")
        processed_dir = paths.processed_data / "v1"
        
        X_train = loader.load_csv(processed_dir / "X_train.csv", validate=False)
        X_test = loader.load_csv(processed_dir / "X_test.csv", validate=False)
        y_train = loader.load_csv(processed_dir / "y_train.csv", validate=False).squeeze()
        y_test = loader.load_csv(processed_dir / "y_test.csv", validate=False).squeeze()
        
        # Log data info
        mlflow_tracker.log_params({
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'n_features': X_train.shape[1]
        })
        
        # Step 2: Initialize model
        logger.info("\n[Step 2/6] Initializing Random Forest model...")
        model = RandomForestModel(**model_params)
        mlflow_tracker.log_params(model_params)
        
        # Step 3: Train model
        logger.info("\n[Step 3/6] Training model...")
        trainer = ModelTrainer(model)
        trainer.train(X_train, y_train)
        
        # Step 4: Cross-validation
        logger.info("\n[Step 4/6] Performing cross-validation...")
        cv_scores = trainer.cross_validate(X_train, y_train, cv=5, scoring='accuracy')
        mlflow_tracker.log_metrics({
            'cv_mean_accuracy': cv_scores['mean'],
            'cv_std_accuracy': cv_scores['std']
        })
        
        # Step 5: Evaluate model
        logger.info("\n[Step 5/6] Evaluating model...")
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)
        mlflow_tracker.log_metrics(metrics)
        
        # Get classification report
        evaluator.get_classification_report(X_test, y_test)
        
        # Calculate business metrics
        business_metrics = evaluator.calculate_business_metrics(X_test, y_test)
        mlflow_tracker.log_metrics(business_metrics)
        
        # Log prediction samples
        comparison = evaluator.compare_predictions(X_test, y_test, sample_size=20)
        
        # Step 6: Save model
        logger.info("\n[Step 6/6] Saving model...")
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = trainer.save_model(version=model_version)
        
        # Log model to MLflow
        mlflow_tracker.log_model(
            model.model,
            artifact_path="model",
            registered_model_name="customer_churn_rf"
        )
        
        # Add model to DVC
        dvc.add(model_path)
        
        # Log model path
        mlflow_tracker.log_params({'model_path': str(model_path)})
        
        # Get run info
        run_info = mlflow_tracker.get_run_info()
        
        logger.info("\n" + "=" * 70)
        logger.info("Model Training Pipeline Completed Successfully!")
        logger.info("=" * 70)
        logger.info(f"\nMLflow Run ID: {run_info['run_id']}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"\nKey Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"\nBusiness Metrics:")
        logger.info(f"  Net Value: ${business_metrics['net_value']:.2f}")
        logger.info(f"  ROI: {business_metrics['roi']:.2f}%")
        
        # End run successfully
        mlflow_tracker.end_run(status="FINISHED")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        mlflow_tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()