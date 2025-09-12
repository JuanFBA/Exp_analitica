"""
ML model training module
"""
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Tuple
import joblib
import os

class MLModel:
    """ML Model wrapper for training and evaluation"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the ML model
        
        Args:
            model_type: Type of model to use ('random_forest' or 'linear_regression')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "linear_regression":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        logging.info(f"Training {self.model_type} model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logging.info("Model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Model predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logging.info("Evaluating model...")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
        
        logging.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        logging.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Test model training
    from data_loader import load_california_housing
    from preprocessing import preprocess_data
    
    logging.basicConfig(level=logging.INFO)
    
    # Load and preprocess data
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train and evaluate model
    model = MLModel("random_forest")
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Model performance: {metrics}")