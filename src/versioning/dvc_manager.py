"""
DVC Manager
Manages DVC operations for data and model versioning.
SRP: Responsible only for DVC-related operations.
"""

import subprocess
from pathlib import Path
from typing import List, Optional
import json

from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class DVCManager:
    """Manages DVC versioning operations."""
    
    def __init__(self):
        """Initialize DVC manager."""
        self.config = config.get_dvc_config()
        self._setup_remote()
    
    def _setup_remote(self) -> None:
        """Setup DVC remote storage."""
        try:
            # Check if remote exists
            result = subprocess.run(
                ['dvc', 'remote', 'list'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if self.config.remote_name not in result.stdout:
                # Add remote
                subprocess.run(
                    ['dvc', 'remote', 'add', '-d', self.config.remote_name, self.config.remote_url],
                    check=True
                )
                logger.info(f"DVC remote '{self.config.remote_name}' added: {self.config.remote_url}")
            else:
                logger.info(f"DVC remote '{self.config.remote_name}' already exists")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to setup DVC remote: {e}")
            raise
    
    def add(self, path: Path) -> bool:
        """
        Add file or directory to DVC tracking.
        
        Args:
            path: Path to add
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Adding to DVC: {path}")
            subprocess.run(['dvc', 'add', str(path)], check=True)
            
            # Auto-stage .dvc file to git if configured
            if self.config.autostage:
                dvc_file = Path(str(path) + '.dvc')
                if dvc_file.exists():
                    subprocess.run(['git', 'add', str(dvc_file)], check=False)
                    logger.info(f"Staged {dvc_file.name} to git")
            
            logger.info(f"Successfully added {path} to DVC")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add {path} to DVC: {e}")
            return False
    
    def push(self, target: Optional[str] = None) -> bool:
        """
        Push data to DVC remote.
        
        Args:
            target: Specific target to push (optional)
            
        Returns:
            True if successful
        """
        try:
            cmd = ['dvc', 'push']
            if target:
                cmd.append(target)
            
            logger.info("Pushing to DVC remote...")
            subprocess.run(cmd, check=True)
            logger.info("Successfully pushed to DVC remote")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push to DVC remote: {e}")
            return False
    
    def pull(self, target: Optional[str] = None) -> bool:
        """
        Pull data from DVC remote.
        
        Args:
            target: Specific target to pull (optional)
            
        Returns:
            True if successful
        """
        try:
            cmd = ['dvc', 'pull']
            if target:
                cmd.append(target)
            
            logger.info("Pulling from DVC remote...")
            subprocess.run(cmd, check=True)
            logger.info("Successfully pulled from DVC remote")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull from DVC remote: {e}")
            return False
    
    def checkout(self, target: str) -> bool:
        """
        Checkout specific version of data.
        
        Args:
            target: Target .dvc file or directory
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Checking out DVC target: {target}")
            subprocess.run(['dvc', 'checkout', target], check=True)
            logger.info(f"Successfully checked out {target}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout {target}: {e}")
            return False
    
    def status(self) -> dict:
        """
        Get DVC status.
        
        Returns:
            Dictionary with status information
        """
        try:
            result = subprocess.run(
                ['dvc', 'status', '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            status = json.loads(result.stdout) if result.stdout else {}
            return status
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get DVC status: {e}")
            return {}
    
    def get_tracked_files(self) -> List[str]:
        """
        Get list of files tracked by DVC.
        
        Returns:
            List of tracked file paths
        """
        try:
            result = subprocess.run(
                ['dvc', 'list', '.', '--dvc-only'],
                capture_output=True,
                text=True,
                check=False
            )
            
            files = result.stdout.strip().split('\n') if result.stdout else []
            return [f for f in files if f]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list DVC files: {e}")
            return []