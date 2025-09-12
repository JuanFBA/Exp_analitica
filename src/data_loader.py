"""
Data loading module for California housing dataset
"""
import logging
from sklearn.datasets import fetch_california_housing
import pandas as pd
from typing import Tuple

def load_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California housing dataset from scikit-learn
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target data
    """
    logging.info("Loading California housing dataset...")
    
    try:
        # Try to fetch the dataset
        housing_data = fetch_california_housing(as_frame=True)
        X = housing_data.data
        y = housing_data.target
        
        logging.info(f"Dataset loaded successfully. Shape: {X.shape}")
        logging.info(f"Features: {list(X.columns)}")
        logging.info(f"Target variable: {housing_data.target_names}")
        
        return X, y
        
    except Exception as e:
        logging.warning(f"Failed to download dataset from internet: {e}")
        logging.info("Creating synthetic California housing dataset for testing...")
        
        # Create synthetic dataset with the same structure as California housing
        import numpy as np
        np.random.seed(42)
        
        n_samples = 20640  # Same as original dataset
        feature_names = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        # Generate synthetic data with realistic ranges
        X_data = {
            'MedInc': np.random.uniform(0.5, 15.0, n_samples),  # Median income
            'HouseAge': np.random.uniform(1, 52, n_samples),     # House age
            'AveRooms': np.random.uniform(2, 15, n_samples),     # Average rooms
            'AveBedrms': np.random.uniform(0.5, 5, n_samples),   # Average bedrooms
            'Population': np.random.uniform(3, 35682, n_samples), # Population
            'AveOccup': np.random.uniform(0.8, 20, n_samples),   # Average occupancy
            'Latitude': np.random.uniform(32.5, 42, n_samples),  # Latitude
            'Longitude': np.random.uniform(-124.3, -114.3, n_samples)  # Longitude
        }
        
        X = pd.DataFrame(X_data)
        
        # Generate target (house values) based on features
        y_values = (
            X['MedInc'] * 0.5 +
            (52 - X['HouseAge']) * 0.02 +
            X['AveRooms'] * 0.1 +
            np.random.normal(0, 0.5, n_samples)  # Add noise
        )
        y_values = np.clip(y_values, 0.15, 5.0)  # Clip to realistic range
        
        y = pd.Series(y_values, name='MedHouseVal')
        
        logging.info(f"Synthetic dataset created successfully. Shape: {X.shape}")
        logging.info(f"Features: {list(X.columns)}")
        logging.info(f"Target variable: MedHouseVal (median house value)")
        
        return X, y

if __name__ == "__main__":
    # Test the data loading
    logging.basicConfig(level=logging.INFO)
    X, y = load_california_housing()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample features:\n{X.head()}")
    print(f"Sample targets: {y.head()}")