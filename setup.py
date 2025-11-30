from setuptools import setup, find_packages

setup(
    name="mlops-versioning-system",
    version="0.1.0",
    description="Production-ready MLOps system with full versioning and rollback capabilities",
    author="Dani",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "dvc>=3.30.0",
        "mlflow>=2.9.2",
        "pyyaml>=6.0.1",
        "python-json-logger>=2.0.7",
        "colorlog>=6.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
        ],
        "api": [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
)