"""
Rollback Script
Quick rollback to a previous version.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.versioning.version_controller import VersionController
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Rollback to a specific version."""
    version_ctrl = VersionController()
    
    # List available snapshots
    logger.info("=" * 70)
    logger.info("Available Snapshots:")
    logger.info("=" * 70)
    
    snapshots = version_ctrl.list_snapshots()
    
    if not snapshots:
        logger.warning("No snapshots found!")
        return
    
    for i, snapshot in enumerate(snapshots, 1):
        logger.info(f"\n{i}. Version: {snapshot['version']}")
        logger.info(f"   Timestamp: {snapshot['timestamp']}")
        logger.info(f"   Description: {snapshot['description']}")
        logger.info(f"   Git Commit: {snapshot.get('git_commit', 'N/A')[:8]}")
    
    # Get user choice
    print("\n" + "=" * 70)
    choice = input("Enter snapshot number to rollback (or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        logger.info("Rollback cancelled")
        return
    
    try:
        snapshot_idx = int(choice) - 1
        if 0 <= snapshot_idx < len(snapshots):
            target_snapshot = snapshots[snapshot_idx]
            
            logger.info(f"\nRolling back to: {target_snapshot['version']}")
            
            # Perform rollback
            success = version_ctrl.restore_snapshot(
                target_snapshot['version'],
                restore_data=True,
                restore_models=True,
                restore_code=True
            )
            
            if success:
                logger.info("\n✅ Rollback completed successfully!")
            else:
                logger.error("\n❌ Rollback failed!")
        else:
            logger.error("Invalid snapshot number!")
            
    except ValueError:
        logger.error("Invalid input!")


if __name__ == "__main__":
    main()