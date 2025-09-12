"""
Data preprocessing module
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def preprocess_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess the housing data
    
    Args:
        X: Feature data
        y: Target data
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and fitted scaler
    """
    logging.info("Starting data preprocessing...")
    
    # Check for missing values
    if X.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in features")
        X = X.fillna(X.mean())
    
    if y.isnull().sum() > 0:
        logging.warning("Missing values detected in target")
        y = y.fillna(y.mean())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logging.info(f"Train set size: {X_train.shape[0]}")
    logging.info(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logging.info("Data preprocessing completed successfully")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_california_housing
    
    logging.basicConfig(level=logging.INFO)
    
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training targets shape: {y_train.shape}")
    print(f"Test targets shape: {y_test.shape}")