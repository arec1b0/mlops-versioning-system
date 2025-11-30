"""
Integration Tests
Tests complete system workflows end-to-end.
"""

import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger
from src.data.data_generator import generate_and_save_data
from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.models.random_forest_model import RandomForestModel
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.versioning.dvc_manager import DVCManager
from src.versioning.mlflow_tracker import MLflowTracker
from src.versioning.version_controller import VersionController
from src.incident.incident_generator import IncidentGenerator
from src.incident.recovery_manager import RecoveryManager

logger = get_logger(__name__)


class IntegrationTester:
    """Runs integration tests for the entire system."""
    
    def __init__(self):
        """Initialize tester."""
        self.paths = config.get_paths()
        self.test_passed = []
        self.test_failed = []
    
    def test_data_pipeline(self):
        """Test complete data pipeline."""
        logger.info("\n[TEST] Data Pipeline")
        
        try:
            # Generate data
            test_data_path = self.paths.raw_data / "test_data.csv"
            df = generate_and_save_data(test_data_path, n_samples=100, n_features=10)
            
            # Load and process
            loader = DataLoader()
            processor = DataProcessor()
            
            df = loader.load_csv(test_data_path)
            X, y = processor.split_features_target(df)
            X_train, X_test, y_train, y_test = processor.train_test_split_data(X, y)
            
            # Clean up
            test_data_path.unlink()
            
            logger.info("✅ Data pipeline test PASSED")
            self.test_passed.append("Data Pipeline")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data pipeline test FAILED: {e}")
            self.test_failed.append("Data Pipeline")
            return False
    
    def test_model_training(self):
        """Test model training workflow."""
        logger.info("\n[TEST] Model Training")
        
        try:
            # Load processed data
            processed_dir = self.paths.processed_data / "v1"
            loader = DataLoader()
            
            X_train = loader.load_csv(processed_dir / "X_train.csv", validate=False)
            y_train = loader.load_csv(processed_dir / "y_train.csv", validate=False).squeeze()
            
            # Train small model
            model = RandomForestModel(n_estimators=10, max_depth=3, random_state=42)
            trainer = ModelTrainer(model)
            trainer.train(X_train[:100], y_train[:100])
            
            assert model.is_fitted, "Model should be fitted"
            
            logger.info("✅ Model training test PASSED")
            self.test_passed.append("Model Training")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training test FAILED: {e}")
            self.test_failed.append("Model Training")
            return False
    
    def test_versioning(self):
        """Test versioning system."""
        logger.info("\n[TEST] Versioning System")
        
        try:
            version_ctrl = VersionController()
            
            # Get current version
            current_version = version_ctrl.get_current_version()
            assert 'git_commit' in current_version, "Should have git commit"
            
            # List snapshots
            snapshots = version_ctrl.list_snapshots()
            assert isinstance(snapshots, list), "Should return list"
            
            logger.info("✅ Versioning test PASSED")
            self.test_passed.append("Versioning")
            return True
            
        except Exception as e:
            logger.error(f"❌ Versioning test FAILED: {e}")
            self.test_failed.append("Versioning")
            return False
    
    def test_incident_recovery(self):
        """Test incident and recovery system."""
        logger.info("\n[TEST] Incident & Recovery")
        
        try:
            incident_gen = IncidentGenerator()
            recovery_mgr = RecoveryManager()
            
            # Simulate pipeline failure (safest test)
            incident = incident_gen.simulate_pipeline_failure('test')
            assert incident['status'] == 'active', "Incident should be active"
            
            # Recover
            result = recovery_mgr.recover_from_incident(incident)
            assert result['status'] == 'success', "Recovery should succeed"
            
            logger.info("✅ Incident & Recovery test PASSED")
            self.test_passed.append("Incident & Recovery")
            return True
            
        except Exception as e:
            logger.error(f"❌ Incident & Recovery test FAILED: {e}")
            self.test_failed.append("Incident & Recovery")
            return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("=" * 70)
        logger.info("RUNNING INTEGRATION TESTS")
        logger.info("=" * 70)
        
        self.test_data_pipeline()
        self.test_model_training()
        self.test_versioning()
        self.test_incident_recovery()
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✅ Passed: {len(self.test_passed)}")
        logger.info(f"❌ Failed: {len(self.test_failed)}")
        
        if self.test_passed:
            logger.info("\nPassed Tests:")
            for test in self.test_passed:
                logger.info(f"  ✅ {test}")
        
        if self.test_failed:
            logger.info("\nFailed Tests:")
            for test in self.test_failed:
                logger.info(f"  ❌ {test}")
        
        logger.info("=" * 70)
        
        return len(self.test_failed) == 0


def main():
    """Run integration tests."""
    tester = IntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("\n🎉 All integration tests PASSED!")
        sys.exit(0)
    else:
        logger.error("\n⚠️  Some integration tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()