"""
Recovery Manager
Handles automated recovery from incidents.
SRP: Responsible only for recovery operations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import shutil
from datetime import datetime

from src.versioning.version_controller import VersionController
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class RecoveryManager:
    """Manages automated recovery from incidents."""
    
    def __init__(self):
        """Initialize recovery manager."""
        self.version_controller = VersionController()
        self.paths = config.get_paths()
        self.recovery_timeout = config.get('incident.recovery_timeout', 300)
    
    def recover_from_incident(
        self,
        incident: Dict[str, Any],
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recover from an incident.
        
        Args:
            incident: Incident information
            target_version: Specific version to restore (None = latest)
            
        Returns:
            Recovery result information
        """
        logger.info("=" * 70)
        logger.info("🚑 STARTING AUTOMATED RECOVERY")
        logger.info("=" * 70)
        
        recovery_result = {
            'incident': incident,
            'start_time': datetime.now().isoformat(),
            'status': 'in_progress',
            'actions_taken': []
        }
        
        try:
            incident_type = incident.get('type')
            
            if incident_type == 'data_corruption':
                recovery_result = self._recover_data_corruption(incident, recovery_result)
                
            elif incident_type == 'model_degradation':
                recovery_result = self._recover_model_degradation(incident, recovery_result)
                
            elif incident_type == 'pipeline_failure':
                recovery_result = self._recover_pipeline_failure(incident, recovery_result)
            
            else:
                logger.warning(f"Unknown incident type: {incident_type}")
                recovery_result['status'] = 'unknown_incident_type'
            
            recovery_result['end_time'] = datetime.now().isoformat()
            
            if recovery_result['status'] == 'success':
                logger.info("=" * 70)
                logger.info("✅ RECOVERY COMPLETED SUCCESSFULLY")
                logger.info("=" * 70)
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}", exc_info=True)
            recovery_result['status'] = 'failed'
            recovery_result['error'] = str(e)
            recovery_result['end_time'] = datetime.now().isoformat()
            return recovery_result
    
    def _recover_data_corruption(
        self,
        incident: Dict[str, Any],
        recovery_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from data corruption."""
        logger.info("Recovering from data corruption...")
        
        affected_file = Path(incident['affected_file'])
        backup_path = Path(incident.get('backup_path', ''))
        
        # Strategy 1: Restore from backup
        if backup_path.exists():
            logger.info(f"Restoring from backup: {backup_path}")
            shutil.copy2(backup_path, affected_file)
            recovery_result['actions_taken'].append('restored_from_backup')
            recovery_result['status'] = 'success'
            logger.info(f"✅ Data restored from backup")
            return recovery_result
        
        # Strategy 2: Restore from DVC
        logger.info("Attempting DVC restore...")
        if self.version_controller.dvc.checkout(str(affected_file) + '.dvc'):
            recovery_result['actions_taken'].append('restored_from_dvc')
            recovery_result['status'] = 'success'
            logger.info(f"✅ Data restored from DVC")
            return recovery_result
        
        # Strategy 3: Full system restore
        logger.warning("Individual file recovery failed, attempting full restore...")
        snapshots = self.version_controller.list_snapshots()
        
        if snapshots:
            latest_snapshot = snapshots[-1]
            logger.info(f"Restoring to snapshot: {latest_snapshot['version']}")
            
            if self.version_controller.restore_snapshot(
                latest_snapshot['version'],
                restore_data=True,
                restore_models=False,
                restore_code=False
            ):
                recovery_result['actions_taken'].append('full_data_restore')
                recovery_result['restored_version'] = latest_snapshot['version']
                recovery_result['status'] = 'success'
                logger.info(f"✅ System restored to: {latest_snapshot['version']}")
                return recovery_result
        
        recovery_result['status'] = 'failed'
        recovery_result['error'] = 'All recovery strategies failed'
        return recovery_result
    
    def _recover_model_degradation(
        self,
        incident: Dict[str, Any],
        recovery_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from model degradation."""
        logger.info("Recovering from model degradation...")
        
        affected_file = Path(incident['affected_file'])
        backup_path = Path(incident.get('backup_path', ''))
        
        # Strategy 1: Restore from backup
        if backup_path.exists():
            logger.info(f"Restoring model from backup: {backup_path}")
            affected_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, affected_file)
            recovery_result['actions_taken'].append('restored_from_backup')
            recovery_result['status'] = 'success'
            logger.info(f"✅ Model restored from backup")
            return recovery_result
        
        # Strategy 2: Restore from DVC
        logger.info("Attempting DVC restore...")
        dvc_file = str(affected_file) + '.dvc'
        if Path(dvc_file).exists() and self.version_controller.dvc.checkout(dvc_file):
            recovery_result['actions_taken'].append('restored_from_dvc')
            recovery_result['status'] = 'success'
            logger.info(f"✅ Model restored from DVC")
            return recovery_result
        
        # Strategy 3: Use MLflow model registry
        logger.info("Attempting MLflow model restore...")
        # This would query MLflow registry and download latest production model
        # For now, we'll use DVC as primary source
        
        recovery_result['status'] = 'failed'
        recovery_result['error'] = 'All recovery strategies failed'
        return recovery_result
    
    def _recover_pipeline_failure(
        self,
        incident: Dict[str, Any],
        recovery_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from pipeline failure."""
        logger.info("Recovering from pipeline failure...")
        
        # Remove failure markers
        marker_file = Path(incident.get('marker_file', ''))
        if marker_file.exists():
            marker_file.unlink()
            logger.info(f"Removed failure marker: {marker_file}")
        
        # For pipeline failures, we typically need to:
        # 1. Clear any partial outputs
        # 2. Reset to known good state
        # 3. Re-run the pipeline
        
        recovery_result['actions_taken'].append('cleared_failure_markers')
        recovery_result['status'] = 'success'
        recovery_result['note'] = 'Pipeline can be restarted'
        
        logger.info("✅ Pipeline failure markers cleared")
        
        return recovery_result
    
    def verify_recovery(self, affected_path: Path) -> bool:
        """
        Verify that recovery was successful.
        
        Args:
            affected_path: Path to verify
            
        Returns:
            True if verification passed
        """
        logger.info(f"Verifying recovery for: {affected_path}")
        
        if not affected_path.exists():
            logger.error(f"Verification failed: File not found: {affected_path}")
            return False
        
        # Check file size
        file_size = affected_path.stat().st_size
        if file_size == 0:
            logger.error(f"Verification failed: File is empty: {affected_path}")
            return False
        
        logger.info(f"✅ Verification passed: {affected_path}")
        return True