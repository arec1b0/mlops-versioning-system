"""
MLflow Tracker
Manages MLflow experiment tracking and model registry.
SRP: Responsible only for MLflow operations.
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
from pathlib import Path
import json

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class MLflowTracker:
    """Manages MLflow tracking and model registry."""
    
    def __init__(self):
        """Initialize MLflow tracker."""
        self.mlflow_config = config.get_mlflow_config()
        self._setup_mlflow()
        self.run_id: Optional[str] = None
        self.experiment_id: Optional[str] = None
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    self.mlflow_config.experiment_name,
                    artifact_location=self.mlflow_config.artifact_location
                )
                logger.info(f"Created MLflow experiment: {self.mlflow_config.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.mlflow_config.experiment_name}")
            
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags
            
        Returns:
            Run ID
        """
        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        
        # Log default tags
        default_tags = {
            'project': config.project_name,
            'version': config.project_version,
        }
        
        if tags:
            default_tags.update(tags)
        
        mlflow.set_tags(default_tags)
        
        logger.info(f"Started MLflow run: {self.run_id}")
        logger.info(f"Run name: {run_name}")
        
        return self.run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Parameters dictionary
        """
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters to MLflow")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Metrics dictionary
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics to MLflow")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log model to MLflow.
        
        Args:
            model: Model object
            artifact_path: Path within artifacts
            registered_model_name: Name for model registry
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged model to MLflow: {artifact_path}")
        
        if registered_model_name:
            logger.info(f"Registered model: {registered_model_name}")
    
    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None:
        """
        Log artifact file to MLflow.
        
        Args:
            local_path: Path to local file
            artifact_path: Path within artifacts
        """
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.info(f"Logged artifact: {local_path.name}")
    
    def log_dict(self, dictionary: Dict, filename: str) -> None:
        """
        Log dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Output filename
        """
        mlflow.log_dict(dictionary, filename)
        logger.info(f"Logged dictionary: {filename}")
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if status != "FINISHED":
            mlflow.end_run(status=status)
            logger.warning(f"Run ended with status: {status}")
        else:
            mlflow.end_run()
            logger.info(f"MLflow run ended: {self.run_id}")
        
        self.run_id = None
    
    def get_run_info(self) -> Dict[str, Any]:
        """
        Get current run information.
        
        Returns:
            Dictionary with run info
        """
        if self.run_id is None:
            return {}
        
        run = mlflow.get_run(self.run_id)
        
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,
        }
    
    def load_model(self, model_uri: str) -> Any:
        """
        Load model from MLflow.
        
        Args:
            model_uri: Model URI (e.g., "runs:/<run_id>/model")
            
        Returns:
            Loaded model
        """
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from: {model_uri}")
        return model
    
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 10
    ) -> list:
        """
        Search MLflow runs.
        
        Args:
            filter_string: Filter query
            max_results: Maximum results
            
        Returns:
            List of runs
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        logger.info(f"Found {len(runs)} runs matching filter")
        return runs.to_dict('records') if not runs.empty else []