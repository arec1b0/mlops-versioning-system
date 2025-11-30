"""
Configuration Manager
Handles loading and accessing configuration from YAML files.
SRP: Single responsibility for configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class PathConfig:
    """Data paths configuration."""
    raw_data: Path
    processed_data: Path
    trained_models: Path
    logs: Path
    mlruns: Path


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str
    experiment_name: str
    registry_uri: str
    artifact_location: str


@dataclass
class DVCConfig:
    """DVC configuration."""
    remote_name: str
    remote_url: str
    autostage: bool


class ConfigManager:
    """
    Manages application configuration.
    Implements Singleton pattern for consistent config access.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        """Ensure single instance (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        project_root = Path(__file__).parent.parent.parent
        
        # Load main config
        config_path = project_root / "config" / "config.yaml"
        mlflow_config_path = project_root / "config" / "mlflow_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Load MLflow config if exists
        if mlflow_config_path.exists():
            with open(mlflow_config_path, 'r') as f:
                mlflow_cfg = yaml.safe_load(f)
                self._config['mlflow'].update(mlflow_cfg.get('mlflow', {}))
        
        # Convert relative paths to absolute
        self._resolve_paths(project_root)
    
    def _resolve_paths(self, project_root: Path) -> None:
        """Convert relative paths to absolute paths."""
        paths = self._config.get('paths', {})
        
        for category, sub_paths in paths.items():
            if isinstance(sub_paths, dict):
                for key, path in sub_paths.items():
                    abs_path = project_root / path
                    abs_path.mkdir(parents=True, exist_ok=True)
                    paths[category][key] = str(abs_path)
            else:
                abs_path = project_root / sub_paths
                abs_path.mkdir(parents=True, exist_ok=True)
                paths[category] = str(abs_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        Supports nested keys with dot notation (e.g., 'paths.data.raw').
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_paths(self) -> PathConfig:
        """Get paths configuration as dataclass."""
        paths = self._config['paths']
        return PathConfig(
            raw_data=Path(paths['data']['raw']),
            processed_data=Path(paths['data']['processed']),
            trained_models=Path(paths['models']['trained']),
            logs=Path(paths['logs']),
            mlruns=Path(paths['mlruns'])
        )
    
    def get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow configuration as dataclass."""
        mlflow = self._config['mlflow']
        return MLflowConfig(
            tracking_uri=mlflow['tracking_uri'],
            experiment_name=mlflow['experiment_name'],
            registry_uri=mlflow['registry_uri'],
            artifact_location=mlflow['artifact_location']
        )
    
    def get_dvc_config(self) -> DVCConfig:
        """Get DVC configuration as dataclass."""
        dvc = self._config['dvc']
        return DVCConfig(
            remote_name=dvc['remote_name'],
            remote_url=dvc['remote_url'],
            autostage=dvc['autostage']
        )
    
    @property
    def project_name(self) -> str:
        """Get project name."""
        return self._config['project']['name']
    
    @property
    def project_version(self) -> str:
        """Get project version."""
        return self._config['project']['version']
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = None
        self._load_config()


# Global config instance
config = ConfigManager()