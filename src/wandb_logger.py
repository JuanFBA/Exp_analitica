"""
WandB logging functionality
"""
import logging
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
import os

class WandBLogger:
    """WandB logging wrapper for ML experiments"""
    
    def __init__(self, project_name: str = "california-housing-ml", entity: Optional[str] = None):
        """
        Initialize WandB logger
        
        Args:
            project_name: WandB project name
            entity: WandB entity (username/team)
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None
        
    def init_run(self, config: Dict[str, Any], run_name: Optional[str] = None) -> None:
        """
        Initialize a WandB run
        
        Args:
            config: Configuration dictionary to log
            run_name: Optional name for the run
        """
        try:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=config,
                name=run_name,
                reinit=True
            )
            logging.info(f"WandB run initialized: {self.run.name}")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}")
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to WandB
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.run is None:
            logging.warning("WandB run not initialized. Skipping metric logging.")
            return
        
        try:
            self.run.log(metrics, step=step)
            logging.info(f"Logged metrics to WandB: {metrics}")
        except Exception as e:
            logging.warning(f"Failed to log metrics to WandB: {e}")
    
    def log_model_performance_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Create and log model performance plots
        
        Args:
            y_true: True values
            y_pred: Predicted values
        """
        if self.run is None:
            logging.warning("WandB run not initialized. Skipping plot logging.")
            return
        
        try:
            # Create prediction vs actual plot
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual Values')
            
            # Create residuals plot
            plt.subplot(2, 2, 2)
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            
            # Create histogram of residuals
            plt.subplot(2, 2, 3)
            plt.hist(residuals, bins=30, alpha=0.7)
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Distribution of Residuals')
            
            # Create Q-Q plot approximation
            plt.subplot(2, 2, 4)
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot of Residuals')
            
            plt.tight_layout()
            
            # Log to WandB
            self.run.log({"model_performance_plots": wandb.Image(plt)})
            plt.close()
            
            logging.info("Model performance plots logged to WandB")
            
        except Exception as e:
            logging.warning(f"Failed to log plots to WandB: {e}")
    
    def log_feature_importance(self, feature_names: list, importance_values: np.ndarray) -> None:
        """
        Log feature importance plot
        
        Args:
            feature_names: List of feature names
            importance_values: Feature importance values
        """
        if self.run is None:
            logging.warning("WandB run not initialized. Skipping feature importance logging.")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Sort features by importance
            sorted_indices = np.argsort(importance_values)[::-1]
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_values = importance_values[sorted_indices]
            
            plt.barh(range(len(sorted_names)), sorted_values)
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Plot')
            plt.tight_layout()
            
            # Log to WandB
            self.run.log({"feature_importance": wandb.Image(plt)})
            plt.close()
            
            logging.info("Feature importance plot logged to WandB")
            
        except Exception as e:
            logging.warning(f"Failed to log feature importance to WandB: {e}")
    
    def finish_run(self) -> None:
        """Finish the WandB run"""
        if self.run is not None:
            try:
                self.run.finish()
                logging.info("WandB run finished")
            except Exception as e:
                logging.warning(f"Failed to finish WandB run: {e}")
    
    def test_connection(self) -> bool:
        """
        Test WandB connection
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to initialize a test run
            test_run = wandb.init(
                project=f"{self.project_name}-test",
                entity=self.entity,
                mode="disabled"  # Don't actually log anything
            )
            test_run.finish()
            logging.info("WandB connection test successful")
            return True
        except Exception as e:
            logging.warning(f"WandB connection test failed: {e}")
            return False

if __name__ == "__main__":
    # Test WandB logging
    logging.basicConfig(level=logging.INFO)
    
    logger = WandBLogger()
    
    # Test connection
    connection_success = logger.test_connection()
    print(f"WandB connection test: {'SUCCESS' if connection_success else 'FAILED'}")
    
    # Test logging with dummy data
    if connection_success:
        config = {
            "model_type": "test",
            "dataset": "california_housing",
            "test_mode": True
        }
        
        logger.init_run(config, run_name="test_run")
        logger.log_metrics({"test_metric": 0.85})
        logger.finish_run()