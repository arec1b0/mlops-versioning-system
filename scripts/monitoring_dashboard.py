"""
Monitoring Dashboard
Displays system status, metrics, and health information.
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.versioning.version_controller import VersionController
from src.versioning.mlflow_tracker import MLflowTracker
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MonitoringDashboard:
    """Displays system monitoring information."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.version_ctrl = VersionController()
        self.mlflow = MLflowTracker()
        self.paths = config.get_paths()
    
    def display_system_status(self):
        """Display overall system status."""
        print("\n" + "=" * 80)
        print("🎯 MLOPS VERSIONING SYSTEM - MONITORING DASHBOARD")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: {config.project_name}")
        print(f"Version: {config.project_version}")
        print("=" * 80)
    
    def display_version_info(self):
        """Display version information."""
        print("\n📦 VERSION INFORMATION")
        print("-" * 80)
        
        version_info = self.version_ctrl.get_current_version()
        
        print(f"Git Commit:    {version_info.get('git_commit', 'N/A')[:12]}")
        print(f"Git Branch:    {version_info.get('git_branch', 'N/A')}")
        print(f"Latest Tag:    {version_info.get('latest_tag', 'N/A')}")
        print(f"DVC Status:    {version_info.get('dvc_status', 'N/A')}")
    
    def display_snapshots(self):
        """Display available snapshots."""
        print("\n📸 AVAILABLE SNAPSHOTS")
        print("-" * 80)
        
        snapshots = self.version_ctrl.list_snapshots()
        
        if not snapshots:
            print("No snapshots found")
        else:
            for snapshot in snapshots[-5:]:  # Show last 5
                print(f"\nVersion:     {snapshot['version']}")
                print(f"Timestamp:   {snapshot['timestamp']}")
                print(f"Description: {snapshot['description']}")
                print(f"Git Commit:  {snapshot.get('git_commit', 'N/A')[:12]}")
    
    def display_mlflow_experiments(self):
        """Display recent MLflow experiments."""
        print("\n🧪 RECENT EXPERIMENTS (MLflow)")
        print("-" * 80)
        
        try:
            runs = self.mlflow.search_runs(max_results=5)
            
            if not runs:
                print("No experiments found")
            else:
                for i, run in enumerate(runs, 1):
                    print(f"\n{i}. Run: {run.get('run_id', 'N/A')[:8]}")
                    print(f"   Status: {run.get('status', 'N/A')}")
                    
                    # Display key metrics
                    metrics_cols = [col for col in run.keys() if col.startswith('metrics.')]
                    if metrics_cols:
                        print("   Metrics:")
                        for col in metrics_cols[:5]:  # Show first 5 metrics
                            metric_name = col.replace('metrics.', '')
                            value = run.get(col)
                            if value is not None:
                                print(f"     - {metric_name}: {value:.4f}")
        except Exception as e:
            print(f"Error loading experiments: {e}")
    
    def display_data_status(self):
        """Display data status."""
        print("\n📊 DATA STATUS")
        print("-" * 80)
        
        # Raw data
        raw_files = list(self.paths.raw_data.glob("*.csv"))
        print(f"Raw Data Files: {len(raw_files)}")
        for file in raw_files[:3]:  # Show first 3
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.2f} MB")
        
        # Processed data
        processed_dirs = list(self.paths.processed_data.glob("*"))
        print(f"\nProcessed Data Versions: {len(processed_dirs)}")
        for dir in processed_dirs[:3]:  # Show first 3
            if dir.is_dir():
                files = list(dir.glob("*.csv"))
                print(f"  - {dir.name}: {len(files)} files")
    
    def display_model_status(self):
        """Display model status."""
        print("\n🤖 MODEL STATUS")
        print("-" * 80)
        
        model_dirs = list(self.paths.trained_models.glob("*"))
        print(f"Model Versions: {len(model_dirs)}")
        
        for dir in model_dirs[-3:]:  # Show last 3
            if dir.is_dir():
                model_files = list(dir.glob("*.joblib"))
                if model_files:
                    model_file = model_files[0]
                    size_kb = model_file.stat().st_size / 1024
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    print(f"\n  Version: {dir.name}")
                    print(f"  Size: {size_kb:.2f} KB")
                    print(f"  Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def display_health_checks(self):
        """Display system health checks."""
        print("\n💚 HEALTH CHECKS")
        print("-" * 80)
        
        checks = {
            'Data Directory': self.paths.raw_data.exists(),
            'Models Directory': self.paths.trained_models.exists(),
            'Logs Directory': self.paths.logs.exists(),
            'MLflow Tracking': Path('mlruns').exists(),
            'DVC Initialized': Path('.dvc').exists(),
            'Git Repository': Path('.git').exists(),
        }
        
        for check_name, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check_name}")
    
    def display_incident_log(self):
        """Display recent incidents if any."""
        print("\n🚨 INCIDENT LOG")
        print("-" * 80)
        
        incident_markers = Path('.incident_markers')
        
        if incident_markers.exists():
            markers = list(incident_markers.glob("*.marker"))
            if markers:
                print(f"Active Incidents: {len(markers)}")
                for marker in markers[:5]:
                    print(f"  - {marker.stem}")
            else:
                print("No active incidents")
        else:
            print("No active incidents")
    
    def run(self):
        """Run complete dashboard."""
        try:
            self.display_system_status()
            self.display_version_info()
            self.display_snapshots()
            self.display_mlflow_experiments()
            self.display_data_status()
            self.display_model_status()
            self.display_health_checks()
            self.display_incident_log()
            
            print("\n" + "=" * 80)
            print("Dashboard generated successfully!")
            print("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}", exc_info=True)


def main():
    """Main entry point."""
    dashboard = MonitoringDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()