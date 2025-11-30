"""Test configuration and logging."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger


def test_config():
    """Test configuration loading."""
    logger = get_logger(__name__)
    
    logger.info("Testing configuration system...")
    
    # Test basic config access
    project_name = config.project_name
    logger.info(f"Project name: {project_name}")
    
    # Test paths
    paths = config.get_paths()
    logger.info(f"Raw data path: {paths.raw_data}")
    logger.info(f"Processed data path: {paths.processed_data}")
    logger.info(f"Models path: {paths.trained_models}")
    
    # Test MLflow config
    mlflow_cfg = config.get_mlflow_config()
    logger.info(f"MLflow tracking URI: {mlflow_cfg.tracking_uri}")
    logger.info(f"MLflow experiment: {mlflow_cfg.experiment_name}")
    
    # Test DVC config
    dvc_cfg = config.get_dvc_config()
    logger.info(f"DVC remote: {dvc_cfg.remote_name}")
    
    # Test nested access
    model_type = config.get('model.type')
    logger.info(f"Model type: {model_type}")
    
    # Test logging levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    logger.info("Configuration test completed successfully!")


if __name__ == "__main__":
    test_config()