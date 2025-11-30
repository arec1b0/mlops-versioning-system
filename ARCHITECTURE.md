# MLOps Versioning System - Architecture Documentation

## Overview

Production-ready MLOps system with comprehensive versioning, incident simulation, and automated recovery capabilities.

## Architecture Principles

### Core Principles
1. **Reproducibility**: Every experiment, data transformation, and model can be reproduced
2. **Observability**: Complete visibility into system state and operations
3. **Reversibility**: Ability to rollback to any previous state
4. **Load Resistance**: Graceful degradation under failures
5. **Rapid Recovery**: Automated recovery from common incidents

### Design Patterns
- **Single Responsibility Principle (SRP)**: Each module has one clear purpose
- **Open/Closed Principle (OCP)**: Extensible without modification
- **Liskov Substitution Principle (LSP)**: BaseModel allows model swapping
- **Dependency Inversion Principle (DIP)**: Depend on abstractions, not concretions

## System Components

### 1. Configuration Layer (`src/utils/`)
- **ConfigManager**: Centralized configuration management (Singleton pattern)
- **LoggerFactory**: Structured logging with rotation and colored output

### 2. Data Layer (`src/data/`)
- **DataGenerator**: Synthetic dataset generation
- **DataLoader**: Data loading with validation
- **DataProcessor**: Feature engineering and preprocessing

### 3. Model Layer (`src/models/`)
- **BaseModel**: Abstract interface for all models (LSP)
- **RandomForestModel**: Concrete implementation
- **ModelTrainer**: Training workflow management
- **ModelEvaluator**: Comprehensive evaluation metrics

### 4. Versioning Layer (`src/versioning/`)
- **DVCManager**: DVC operations for data/model versioning
- **MLflowTracker**: Experiment tracking and model registry
- **VersionController**: Unified version control coordinator

### 5. Incident Management (`src/incident/`)
- **IncidentGenerator**: Simulates various failure scenarios
- **RecoveryManager**: Automated recovery strategies

## Data Flow
```
Raw Data Generation
    ↓
[DVC Tracking]
    ↓
Data Processing → Feature Engineering
    ↓
[DVC Versioning]
    ↓
Model Training → [MLflow Tracking]
    ↓
Model Evaluation → Metrics Logging
    ↓
Model Storage → [DVC + MLflow Registry]
    ↓
Deployment Ready
```

## Versioning Strategy

### Three-Layer Versioning
1. **Code**: Git for source code
2. **Data/Models**: DVC for large artifacts
3. **Experiments**: MLflow for metrics and parameters

### Snapshot System
- Creates point-in-time snapshots of entire system state
- Includes: code commit, DVC files, MLflow run ID
- Enables complete rollback capability

## Recovery Strategies

### Data Corruption Recovery
1. **Primary**: Restore from local backup
2. **Secondary**: Restore from DVC remote
3. **Tertiary**: Full system restore to last snapshot

### Model Degradation Recovery
1. **Primary**: Restore from local backup
2. **Secondary**: Restore from DVC remote
3. **Tertiary**: Query MLflow registry for production model

### Pipeline Failure Recovery
1. Clear failure markers
2. Reset to known good state
3. Re-run pipeline from failure point

## Key Features

### Incident Simulation
- Data corruption (random noise, missing values, duplicates)
- Model degradation (deletion, corruption)
- Pipeline failures (preprocessing, training, evaluation)

### Automated Recovery
- Multi-strategy recovery approach
- Verification after recovery
- Detailed logging of all actions

### Observability
- Structured logging with rotation
- MLflow experiment tracking
- Business metrics calculation
- Confusion matrix and classification reports

## Technology Stack

- **Python 3.10+**: Core language
- **DVC 3.30+**: Data/model versioning
- **MLflow 2.9+**: Experiment tracking
- **Scikit-learn 1.3+**: ML framework
- **Pandas/Numpy**: Data manipulation
- **Git**: Source control

## Performance Metrics

### Training Pipeline
- Training time: ~0.2s for 4000 samples
- Cross-validation: 5-fold in ~6s
- Model size: ~50KB (Random Forest, 100 trees)

### Recovery Operations
- Backup restore: <1s
- DVC restore: <5s
- Full system restore: <30s

## Security Considerations

1. **Data Access**: All data operations logged
2. **Model Integrity**: Checksums via DVC
3. **Audit Trail**: Complete versioning history
4. **Backup Strategy**: Multiple recovery layers

## Scalability

### Current Capacity
- Data: Up to 1M rows efficiently
- Models: Any scikit-learn compatible model
- Experiments: Unlimited via MLflow

### Extension Points
- Add new model types via BaseModel interface
- Add new incident types via IncidentGenerator
- Add new recovery strategies via RecoveryManager

## Future Enhancements

1. **Distributed Training**: Integration with Ray/Dask
2. **Real-time Monitoring**: Prometheus + Grafana
3. **A/B Testing**: Multi-model deployment
4. **Feature Store**: Centralized feature management
5. **Model Serving**: FastAPI inference endpoints
6. **CI/CD Pipeline**: Automated testing and deployment

## Usage Examples

### Create Snapshot
```python
from src.versioning.version_controller import VersionController

vc = VersionController()
snapshot = vc.create_snapshot(
    version_tag='prod-v1.2.0',
    description='Production release with improved accuracy',
    include_data=True,
    include_models=True
)
```

### Simulate Incident and Recover
```python
from src.incident.incident_generator import IncidentGenerator
from src.incident.recovery_manager import RecoveryManager

# Simulate incident
incident_gen = IncidentGenerator()
incident = incident_gen.simulate_data_corruption(data_file, 'random')

# Recover
recovery_mgr = RecoveryManager()
result = recovery_mgr.recover_from_incident(incident)
```

### Train Model with Tracking
```bash
python scripts/train_pipeline.py
```

### Rollback System
```bash
python scripts/rollback.py
```

## Monitoring and Alerts

### Key Metrics to Monitor
- Data quality (missing values, outliers)
- Model performance (accuracy, F1, ROC-AUC)
- Training time and resource usage
- Recovery success rate
- System availability

### Alert Thresholds
- Accuracy drop > 5%: WARNING
- Accuracy drop > 10%: CRITICAL
- Recovery failure: CRITICAL
- Data corruption detected: WARNING

## Contact and Support

- Documentation: See README.md
- Issues: Create GitHub issue
- Questions: Check documentation first

---

**Last Updated**: 2025-11-30
**Version**: 1.0.0
**Maintained By**: D.Krizhanovskyi