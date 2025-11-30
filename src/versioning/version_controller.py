"""
Version Controller
Unified version control for data, models, and code.
SRP: Responsible for coordinating versioning operations.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from src.versioning.dvc_manager import DVCManager
from src.versioning.mlflow_tracker import MLflowTracker
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class VersionController:
    """
    Unified controller for all versioning operations.
    Coordinates DVC, MLflow, and Git.
    """
    
    def __init__(self):
        """Initialize version controller."""
        self.dvc = DVCManager()
        self.mlflow = MLflowTracker()
        self.paths = config.get_paths()
    
    def create_snapshot(
        self,
        version_tag: str,
        description: str,
        include_data: bool = True,
        include_models: bool = True
    ) -> Dict[str, Any]:
        """
        Create a complete snapshot of current state.
        
        Args:
            version_tag: Version tag (e.g., "v1.2.0")
            description: Description of this version
            include_data: Whether to include data in snapshot
            include_models: Whether to include models in snapshot
            
        Returns:
            Dictionary with snapshot information
        """
        logger.info(f"Creating snapshot: {version_tag}")
        logger.info(f"Description: {description}")
        
        snapshot_info = {
            'version': version_tag,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'git_commit': None,
            'dvc_files': [],
            'mlflow_run_id': None
        }
        
        try:
            # Stage all DVC files
            if include_data:
                logger.info("Adding data files to snapshot...")
                for data_file in self.paths.raw_data.glob("*.csv"):
                    if self.dvc.add(data_file):
                        snapshot_info['dvc_files'].append(str(data_file))
                
                for processed_dir in self.paths.processed_data.glob("*"):
                    if processed_dir.is_dir():
                        for file in processed_dir.glob("*.csv"):
                            if self.dvc.add(file):
                                snapshot_info['dvc_files'].append(str(file))
            
            if include_models:
                logger.info("Adding model files to snapshot...")
                for model_dir in self.paths.trained_models.glob("*"):
                    if model_dir.is_dir():
                        for file in model_dir.glob("*.joblib"):
                            if self.dvc.add(file):
                                snapshot_info['dvc_files'].append(str(file))
            
            # Commit to Git
            logger.info("Committing to Git...")
            self._git_commit(version_tag, description)
            
            # Get git commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            snapshot_info['git_commit'] = result.stdout.strip()
            
            # Create git tag
            self._git_tag(version_tag, description)
            
            # Push to DVC remote
            logger.info("Pushing to DVC remote...")
            self.dvc.push()
            
            # Save snapshot metadata
            self._save_snapshot_metadata(snapshot_info)
            
            logger.info(f"Snapshot created successfully: {version_tag}")
            logger.info(f"Git commit: {snapshot_info['git_commit'][:8]}")
            logger.info(f"DVC files tracked: {len(snapshot_info['dvc_files'])}")
            
            return snapshot_info
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            raise
    
    def restore_snapshot(
        self,
        version_tag: str,
        restore_data: bool = True,
        restore_models: bool = True,
        restore_code: bool = True
    ) -> bool:
        """
        Restore system to a specific snapshot.
        
        Args:
            version_tag: Version tag to restore
            restore_data: Whether to restore data
            restore_models: Whether to restore models
            restore_code: Whether to restore code
            
        Returns:
            True if successful
        """
        logger.info("=" * 70)
        logger.info(f"Starting restore to version: {version_tag}")
        logger.info("=" * 70)
        
        try:
            # Load snapshot metadata
            snapshot_info = self._load_snapshot_metadata(version_tag)
            if not snapshot_info:
                raise ValueError(f"Snapshot not found: {version_tag}")
            
            logger.info(f"Found snapshot from: {snapshot_info['timestamp']}")
            logger.info(f"Description: {snapshot_info['description']}")
            
            # Restore code (git checkout)
            if restore_code:
                logger.info("Restoring code...")
                self._git_checkout(version_tag)
            
            # Restore data and models (dvc checkout)
            if restore_data or restore_models:
                logger.info("Restoring data and models from DVC...")
                # First pull from remote
                self.dvc.pull()
                # Then checkout
                self.dvc.checkout('.')
            
            logger.info("=" * 70)
            logger.info(f"Successfully restored to version: {version_tag}")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available snapshots.
        
        Returns:
            List of snapshot information
        """
        snapshots = []
        
        # List git tags
        try:
            result = subprocess.run(
                ['git', 'tag', '-l'],
                capture_output=True,
                text=True,
                check=True
            )
            
            tags = result.stdout.strip().split('\n')
            
            for tag in tags:
                if tag:
                    snapshot_info = self._load_snapshot_metadata(tag)
                    if snapshot_info:
                        snapshots.append(snapshot_info)
            
            logger.info(f"Found {len(snapshots)} snapshots")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list snapshots: {e}")
        
        return snapshots
    
    def get_current_version(self) -> Dict[str, str]:
        """
        Get current version information.
        
        Returns:
            Dictionary with current version info
        """
        version_info = {}
        
        try:
            # Git commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            version_info['git_commit'] = result.stdout.strip()
            
            # Git branch
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            version_info['git_branch'] = result.stdout.strip()
            
            # Latest tag
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                capture_output=True,
                text=True,
                check=False
            )
            version_info['latest_tag'] = result.stdout.strip() if result.returncode == 0 else 'N/A'
            
            # DVC status
            dvc_status = self.dvc.status()
            version_info['dvc_status'] = 'clean' if not dvc_status else 'modified'
            
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
        
        return version_info
    
    def _git_commit(self, tag: str, message: str) -> None:
        """Commit changes to git."""
        try:
            # Add all .dvc files
            subprocess.run(['git', 'add', '*.dvc', '.dvc/'], check=False)
            subprocess.run(['git', 'add', '.gitignore'], check=False)
            
            # Commit
            subprocess.run(
                ['git', 'commit', '-m', f'{tag}: {message}'],
                check=False
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git commit warning: {e}")
    
    def _git_tag(self, tag: str, message: str) -> None:
        """Create git tag."""
        try:
            subprocess.run(
                ['git', 'tag', '-a', tag, '-m', message],
                check=True
            )
            logger.info(f"Created git tag: {tag}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git tag already exists or failed: {e}")
    
    def _git_checkout(self, tag: str) -> None:
        """Checkout git tag."""
        subprocess.run(['git', 'checkout', tag], check=True)
    
    def _save_snapshot_metadata(self, snapshot_info: Dict[str, Any]) -> None:
        """Save snapshot metadata to file."""
        metadata_dir = Path('.snapshots')
        metadata_dir.mkdir(exist_ok=True)
        
        metadata_file = metadata_dir / f"{snapshot_info['version']}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(snapshot_info, f, indent=2)
        
        logger.info(f"Saved snapshot metadata: {metadata_file}")
    
    def _load_snapshot_metadata(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """Load snapshot metadata from file."""
        metadata_file = Path('.snapshots') / f"{version_tag}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            return json.load(f)