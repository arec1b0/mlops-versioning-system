"""
Incident Generator
Simulates various types of incidents for testing recovery.
SRP: Responsible only for incident simulation.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import shutil
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class IncidentGenerator:
    """Generates various types of incidents for testing."""
    
    def __init__(self):
        """Initialize incident generator."""
        self.paths = config.get_paths()
        self.incident_log: List[Dict[str, Any]] = []
    
    def simulate_data_corruption(
        self,
        file_path: Path,
        corruption_type: str = 'random'
    ) -> Dict[str, Any]:
        """
        Simulate data corruption.
        
        Args:
            file_path: Path to data file
            corruption_type: Type of corruption (random, missing, duplicate)
            
        Returns:
            Incident information
        """
        logger.warning(f"🔥 SIMULATING DATA CORRUPTION: {corruption_type}")
        
        incident = {
            'type': 'data_corruption',
            'subtype': corruption_type,
            'timestamp': datetime.now().isoformat(),
            'affected_file': str(file_path),
            'status': 'created'
        }
        
        try:
            # Backup original file
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            incident['backup_path'] = str(backup_path)
            
            # Load data
            df = pd.read_csv(file_path)
            original_shape = df.shape
            
            if corruption_type == 'random':
                # Introduce random noise
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    mask = np.random.random(len(df)) < 0.1  # 10% corruption
                    df.loc[mask, col] = np.random.randn(mask.sum()) * 1000
                
            elif corruption_type == 'missing':
                # Introduce missing values
                mask = np.random.random(df.shape) < 0.2  # 20% missing
                df = df.mask(mask)
                
            elif corruption_type == 'duplicate':
                # Introduce duplicates
                n_duplicates = int(len(df) * 0.3)
                duplicate_indices = np.random.choice(len(df), n_duplicates)
                duplicates = df.iloc[duplicate_indices]
                df = pd.concat([df, duplicates], ignore_index=True)
            
            # Save corrupted data
            df.to_csv(file_path, index=False)
            
            incident['corrupted_shape'] = df.shape
            incident['original_shape'] = original_shape
            incident['status'] = 'active'
            
            logger.warning(f"Data corrupted: {original_shape} -> {df.shape}")
            logger.warning(f"Backup saved to: {backup_path}")
            
            self.incident_log.append(incident)
            return incident
            
        except Exception as e:
            logger.error(f"Failed to simulate data corruption: {e}")
            incident['status'] = 'failed'
            incident['error'] = str(e)
            return incident
    
    def simulate_model_degradation(
        self,
        model_path: Path,
        degradation_type: str = 'delete'
    ) -> Dict[str, Any]:
        """
        Simulate model degradation or loss.
        
        Args:
            model_path: Path to model file
            degradation_type: Type of degradation (delete, corrupt)
            
        Returns:
            Incident information
        """
        logger.warning(f"🔥 SIMULATING MODEL DEGRADATION: {degradation_type}")
        
        incident = {
            'type': 'model_degradation',
            'subtype': degradation_type,
            'timestamp': datetime.now().isoformat(),
            'affected_file': str(model_path),
            'status': 'created'
        }
        
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Backup original model
            backup_path = model_path.with_suffix(model_path.suffix + '.backup')
            shutil.copy2(model_path, backup_path)
            incident['backup_path'] = str(backup_path)
            
            if degradation_type == 'delete':
                # Delete model file
                model_path.unlink()
                logger.warning(f"Model deleted: {model_path}")
                
            elif degradation_type == 'corrupt':
                # Corrupt model file
                with open(model_path, 'wb') as f:
                    f.write(b'CORRUPTED_DATA' * 1000)
                logger.warning(f"Model corrupted: {model_path}")
            
            incident['status'] = 'active'
            self.incident_log.append(incident)
            return incident
            
        except Exception as e:
            logger.error(f"Failed to simulate model degradation: {e}")
            incident['status'] = 'failed'
            incident['error'] = str(e)
            return incident
    
    def simulate_pipeline_failure(
        self,
        failure_point: str = 'preprocessing'
    ) -> Dict[str, Any]:
        """
        Simulate pipeline failure.
        
        Args:
            failure_point: Point of failure (preprocessing, training, evaluation)
            
        Returns:
            Incident information
        """
        logger.warning(f"🔥 SIMULATING PIPELINE FAILURE: {failure_point}")
        
        incident = {
            'type': 'pipeline_failure',
            'subtype': failure_point,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }
        
        # Simulate by creating a marker file
        marker_file = Path('.incident_markers') / f"pipeline_failure_{failure_point}.marker"
        marker_file.parent.mkdir(exist_ok=True)
        
        with open(marker_file, 'w') as f:
            f.write(f"Pipeline failure at: {failure_point}\n")
            f.write(f"Timestamp: {incident['timestamp']}\n")
        
        incident['marker_file'] = str(marker_file)
        
        logger.warning(f"Pipeline failure marker created: {marker_file}")
        
        self.incident_log.append(incident)
        return incident
    
    def simulate_random_incident(self) -> Dict[str, Any]:
        """
        Simulate a random incident.
        
        Returns:
            Incident information
        """
        incident_types = [
            ('data_corruption', 'random'),
            ('data_corruption', 'missing'),
            ('model_degradation', 'delete'),
            ('pipeline_failure', 'preprocessing')
        ]
        
        incident_type, subtype = random.choice(incident_types)
        
        logger.warning(f"🎲 Generating random incident: {incident_type} - {subtype}")
        
        if incident_type == 'data_corruption':
            data_files = list(self.paths.raw_data.glob("*.csv"))
            if data_files:
                target_file = random.choice(data_files)
                return self.simulate_data_corruption(target_file, subtype)
        
        elif incident_type == 'model_degradation':
            model_files = list(self.paths.trained_models.glob("*/*.joblib"))
            if model_files:
                target_file = random.choice(model_files)
                return self.simulate_model_degradation(target_file, subtype)
        
        elif incident_type == 'pipeline_failure':
            return self.simulate_pipeline_failure(subtype)
        
        return {'type': 'none', 'status': 'no_target_found'}
    
    def get_incident_log(self) -> List[Dict[str, Any]]:
        """Get all incidents in log."""
        return self.incident_log
    
    def clear_incident_log(self) -> None:
        """Clear incident log."""
        self.incident_log = []
        logger.info("Incident log cleared")