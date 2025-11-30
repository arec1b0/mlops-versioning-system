"""
Data Generator
Generates synthetic dataset for demonstration purposes.
SRP: Responsible only for data generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataGenerator:
    """Generates synthetic classification dataset."""
    
    def __init__(self, n_samples: int = 1000, n_features: int = 20, random_state: int = 42):
        """
        Initialize data generator.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_classification_data(self) -> pd.DataFrame:
        """
        Generate synthetic classification dataset.
        Simulates customer churn prediction scenario.
        
        Returns:
            DataFrame with features and target
        """
        logger.info(f"Generating classification dataset: {self.n_samples} samples, {self.n_features} features")
        
        # Generate base features
        data = {}
        
        # Numerical features
        data['age'] = np.random.randint(18, 80, self.n_samples)
        data['account_age_months'] = np.random.randint(1, 120, self.n_samples)
        data['monthly_charges'] = np.random.uniform(20, 200, self.n_samples)
        data['total_charges'] = data['account_age_months'] * data['monthly_charges'] * np.random.uniform(0.8, 1.2, self.n_samples)
        data['support_calls'] = np.random.poisson(2, self.n_samples)
        data['days_since_last_login'] = np.random.exponential(10, self.n_samples)
        
        # Categorical features (encoded)
        data['contract_type'] = np.random.choice([0, 1, 2], self.n_samples, p=[0.3, 0.5, 0.2])  # Month-to-month, One year, Two year
        data['payment_method'] = np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.3, 0.3, 0.2, 0.2])
        data['internet_service'] = np.random.choice([0, 1, 2], self.n_samples, p=[0.2, 0.4, 0.4])  # No, DSL, Fiber
        
        # Additional random features
        for i in range(self.n_features - len(data)):
            data[f'feature_{i}'] = np.random.randn(self.n_samples)
        
        # Generate target with realistic dependencies
        churn_probability = (
            0.1 +  # base probability
            0.3 * (data['support_calls'] > 5) +  # high support calls
            0.2 * (data['days_since_last_login'] > 30) +  # inactive users
            0.15 * (data['contract_type'] == 0) +  # month-to-month contracts
            0.1 * (data['monthly_charges'] > 150) +  # high charges
            -0.2 * (data['account_age_months'] > 60)  # loyal customers
        )
        
        churn_probability = np.clip(churn_probability, 0, 1)
        data['target'] = np.random.binomial(1, churn_probability)
        
        df = pd.DataFrame(data)
        
        # Add metadata
        df['created_at'] = datetime.now()
        df['data_version'] = '1.0.0'
        
        logger.info(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def generate_time_series_versions(self, base_df: pd.DataFrame, n_versions: int = 3) -> list:
        """
        Generate multiple versions of dataset to simulate data drift.
        
        Args:
            base_df: Base DataFrame
            n_versions: Number of versions to create
            
        Returns:
            List of DataFrames with different versions
        """
        versions = []
        
        for version in range(1, n_versions + 1):
            df = base_df.copy()
            
            # Simulate data drift
            drift_factor = 1 + (version * 0.1)
            df['monthly_charges'] = df['monthly_charges'] * drift_factor
            df['support_calls'] = df['support_calls'] + np.random.poisson(version, len(df))
            
            # Update metadata
            df['data_version'] = f'1.{version}.0'
            df['created_at'] = datetime.now() + timedelta(days=version * 30)
            
            versions.append(df)
            logger.info(f"Generated data version {version}: shape {df.shape}")
        
        return versions


def generate_and_save_data(output_path: Path, n_samples: int = 1000, n_features: int = 20):
    """
    Generate and save dataset to specified path.
    
    Args:
        output_path: Path to save the data
        n_samples: Number of samples
        n_features: Number of features
    """
    generator = DataGenerator(n_samples=n_samples, n_features=n_features)
    df = generator.generate_classification_data()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Data saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return df


if __name__ == "__main__":
    from src.utils.config import config
    
    paths = config.get_paths()
    output_file = paths.raw_data / "customer_data.csv"
    
    df = generate_and_save_data(output_file, n_samples=5000, n_features=20)
    print(f"\nDataset preview:\n{df.head()}")
    print(f"\nDataset info:\n{df.info()}")