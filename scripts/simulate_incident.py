"""
Incident Simulation Script
Simulates incidents and tests automated recovery.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger
from src.incident.incident_generator import IncidentGenerator
from src.incident.recovery_manager import RecoveryManager
from src.versioning.version_controller import VersionController

logger = get_logger(__name__)


def main():
    """Run incident simulation and recovery test."""
    logger.info("=" * 70)
    logger.info("INCIDENT SIMULATION AND RECOVERY TEST")
    logger.info("=" * 70)
    
    # Initialize components
    incident_gen = IncidentGenerator()
    recovery_mgr = RecoveryManager()
    version_ctrl = VersionController()
    paths = config.get_paths()
    
    # Step 1: Create a snapshot before incident
    logger.info("\n[Step 1/4] Creating pre-incident snapshot...")
    snapshot = version_ctrl.create_snapshot(
        version_tag='pre-incident-test',
        description='Snapshot before incident simulation',
        include_data=True,
        include_models=True
    )
    
    time.sleep(2)
    
    # Step 2: Simulate an incident
    logger.info("\n[Step 2/4] Simulating incident...")
    
    # Choose incident type
    print("\nSelect incident type to simulate:")
    print("1. Data Corruption (random noise)")
    print("2. Data Corruption (missing values)")
    print("3. Model Degradation (delete)")
    print("4. Pipeline Failure")
    print("5. Random Incident")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        data_file = paths.raw_data / "customer_data.csv"
        incident = incident_gen.simulate_data_corruption(data_file, 'random')
    elif choice == '2':
        data_file = paths.raw_data / "customer_data.csv"
        incident = incident_gen.simulate_data_corruption(data_file, 'missing')
    elif choice == '3':
        model_files = list(paths.trained_models.glob("*/*.joblib"))
        if model_files:
            incident = incident_gen.simulate_model_degradation(model_files[0], 'delete')
        else:
            logger.error("No model files found!")
            return
    elif choice == '4':
        incident = incident_gen.simulate_pipeline_failure('preprocessing')
    elif choice == '5':
        incident = incident_gen.simulate_random_incident()
    else:
        logger.error("Invalid choice!")
        return
    
    logger.info(f"\nIncident created: {incident['type']} - {incident.get('subtype', 'N/A')}")
    time.sleep(2)
    
    # Step 3: Attempt recovery
    logger.info("\n[Step 3/4] Attempting automated recovery...")
    recovery_result = recovery_mgr.recover_from_incident(incident)
    
    logger.info(f"\nRecovery status: {recovery_result['status']}")
    logger.info(f"Actions taken: {recovery_result['actions_taken']}")
    
    time.sleep(2)
    
    # Step 4: Verify recovery
    logger.info("\n[Step 4/4] Verifying recovery...")
    
    if incident['type'] in ['data_corruption', 'model_degradation']:
        affected_path = Path(incident['affected_file'])
        verification = recovery_mgr.verify_recovery(affected_path)
        
        if verification:
            logger.info("✅ Recovery verification PASSED")
        else:
            logger.error("❌ Recovery verification FAILED")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INCIDENT SIMULATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Incident Type: {incident['type']}")
    logger.info(f"Recovery Status: {recovery_result['status']}")
    logger.info(f"Recovery Time: {recovery_result.get('end_time', 'N/A')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()