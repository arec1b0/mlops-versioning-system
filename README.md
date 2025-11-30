# MLOps Versioning System

Production-ready MLOps system with full versioning, incident simulation, and automated recovery capabilities.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/DVC-3.30+-orange.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-blue.svg)](https://mlflow.org/)

## 🎯 Key Features

- **Complete Versioning**: Git + DVC + MLflow for code, data, and experiments
- **Incident Simulation**: Test system resilience with various failure scenarios
- **Automated Recovery**: Multi-strategy recovery from data corruption, model loss, and pipeline failures
- **Production Ready**: Comprehensive logging, monitoring, and observability
- **Reproducibility**: Every experiment is fully reproducible
- **Reversibility**: Rollback to any previous state in seconds

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- Windows 11 (or Linux/Mac with minor path adjustments)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd mlops-versioning-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Initialize DVC
dvc init
```

### Generate and Process Data
```bash
python scripts/prepare_data.py
```

### Train Model
```bash
python scripts/train_pipeline.py
```

### View Experiments in MLflow
```bash
mlflow ui --backend-store-uri mlruns
# Open browser: http://localhost:5000
```

## 📊 System Capabilities

### Data Management

- Synthetic data generation with realistic distributions
- Automated data validation
- Feature engineering and scaling
- DVC-based versioning
- Multiple data versions support

### Model Training

- Random Forest classifier (extensible to other models)
- 5-fold cross-validation
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Business metrics (ROI, net value)
- MLflow experiment tracking

### Versioning

- **Code**: Git version control
- **Data/Models**: DVC with remote storage
- **Experiments**: MLflow tracking
- **Snapshots**: Point-in-time system state capture

### Incident Management

Simulate and recover from:
- Data corruption (random noise, missing values, duplicates)
- Model degradation (deletion, corruption)
- Pipeline failures (preprocessing, training, evaluation)

## 🔧 Usage Examples

### Create System Snapshot
```python
from src.versioning.version_controller import VersionController

vc = VersionController()
snapshot = vc.create_snapshot(
    version_tag='v1.0.0',
    description='Initial production release',
    include_data=True,
    include_models=True
)
```

### Simulate Incident
```bash
python scripts/simulate_incident.py
```

Follow the interactive prompts to:
1. Choose incident type
2. Observe automated recovery
3. Verify recovery success

### Rollback System
```bash
python scripts/rollback.py
```

Select from available snapshots to restore complete system state.

### Load Specific Data Version
```python
from src.versioning.dvc_manager import DVCManager

dvc = DVCManager()
dvc.checkout('data/raw/customer_data.csv.dvc')
```

## 📁 Project Structure
```
mlops-versioning-system/
├── config/                    # Configuration files
│   ├── config.yaml
│   └── mlflow_config.yaml
├── data/                      # Data storage (DVC tracked)
│   ├── raw/
│   └── processed/
├── models/                    # Model storage (DVC tracked)
│   └── trained/
├── src/                       # Source code
│   ├── data/                  # Data layer
│   ├── models/                # Model layer
│   ├── versioning/            # Versioning layer
│   ├── incident/              # Incident management
│   └── utils/                 # Utilities
├── scripts/                   # Executable scripts
│   ├── prepare_data.py
│   ├── train_pipeline.py
│   ├── simulate_incident.py
│   └── rollback.py
├── tests/                     # Tests
├── mlruns/                    # MLflow artifacts
├── logs/                      # Application logs
└── notebooks/                 # Jupyter notebooks
```

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

### Key Components

- **Configuration Layer**: Centralized config management
- **Data Layer**: Loading, processing, validation
- **Model Layer**: Training, evaluation, prediction
- **Versioning Layer**: DVC, MLflow, Git integration
- **Incident Layer**: Simulation and recovery

### Design Principles

- **SRP**: Single Responsibility Principle
- **OCP**: Open/Closed Principle
- **LSP**: Liskov Substitution Principle
- **DIP**: Dependency Inversion Principle

## 📈 Performance Metrics

### Current System Performance

- **Training**: 4000 samples in ~0.2s
- **Cross-validation**: 5-fold in ~6s
- **Model accuracy**: ~89% (synthetic data)
- **Recovery time**: <5s for most scenarios

### Business Metrics

- Net value calculation
- ROI computation
- Cost/revenue analysis
- Confusion matrix insights

## 🔄 CI/CD Integration

### Recommended Workflow

1. **Development**: Local testing with snapshots
2. **Staging**: Automated testing and validation
3. **Production**: Deployment with rollback capability

### Example GitHub Actions (Future)
```yaml
name: ML Pipeline
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/
      - name: Train model
        run: python scripts/train_pipeline.py
```

## 🛡️ Recovery Strategies

### Multi-Layer Recovery

1. **Layer 1**: Local backups (instant)
2. **Layer 2**: DVC remote (seconds)
3. **Layer 3**: Full snapshot restore (seconds to minutes)

### Verification

All recoveries include automated verification:
- File existence check
- Size validation
- Basic integrity checks

## 📊 Monitoring

### MLflow Tracking

- All experiments logged automatically
- Compare runs in UI
- Model registry for production models

### Logging

- Structured logging with rotation
- Colored console output
- File-based logs in `logs/`

### Metrics

- Training metrics
- Evaluation metrics
- Business metrics
- System metrics

## 🧪 Testing

### Run Tests
```bash
# Test configuration
python tests/test_config.py

# Run all tests (when implemented)
pytest tests/
```

### Incident Testing
```bash
python scripts/simulate_incident.py
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes following design principles
4. Add tests
5. Submit pull request

### Code Style

- Follow PEP 8
- Use type hints
- Document all public APIs
- Keep functions focused (SRP)

## 📝 Configuration

### Main Config (`config/config.yaml`)

- Project settings
- Path configurations
- Model hyperparameters
- Logging settings

### MLflow Config (`config/mlflow_config.yaml`)

- Tracking URI
- Experiment name
- Registry settings

## 🐛 Troubleshooting

### Common Issues

**Issue**: DVC push fails
```bash
# Check remote configuration
dvc remote list
# Reconfigure if needed
dvc remote add -d myremote /path/to/remote
```

**Issue**: MLflow UI not accessible
```bash
# Check if running
mlflow ui --backend-store-uri mlruns --port 5000
```

**Issue**: Import errors
```bash
# Reinstall in editable mode
pip install -e .
```

## 📚 Documentation

- [Architecture Documentation](ARCHITECTURE.md)
- [API Documentation](docs/api.md) (Future)
- [User Guide](docs/user_guide.md) (Future)

## 🔮 Roadmap

### Version 1.1
- [ ] Additional model types (XGBoost, LightGBM)
- [ ] Real-time monitoring dashboard
- [ ] Automated alerting system

### Version 1.2
- [ ] Distributed training support
- [ ] Feature store integration
- [ ] A/B testing framework

### Version 2.0
- [ ] Model serving API
- [ ] Kubernetes deployment
- [ ] Multi-cloud support

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

**Dani** - MLOps Lead at ETH Zurich

## 🙏 Acknowledgments

- Anthropic for Claude AI assistance
- DVC team for versioning tools
- MLflow team for experiment tracking
- scikit-learn community

## 📞 Support

For questions or issues:
1. Check [ARCHITECTURE.md](ARCHITECTURE.md)
2. Review [Troubleshooting](#troubleshooting)
3. Create an issue on GitHub

---

**Built with ❤️ for production MLOps**