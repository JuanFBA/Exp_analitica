"""
Main ML pipeline orchestration script
"""
import logging
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_california_housing
from src.preprocessing import preprocess_data
from src.model import MLModel
from src.wandb_logger import WandBLogger

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ml_pipeline.log')
        ]
    )

def main():
    """Main ML pipeline function"""
    setup_logging()
    
    logging.info("Starting ML pipeline for California housing dataset")
    
    # Configuration
    config = {
        "dataset": "california_housing",
        "model_type": "random_forest",
        "test_size": 0.2,
        "random_state": 42,
        "timestamp": datetime.now().isoformat()
    }
    
    # Initialize WandB logger
    wandb_logger = WandBLogger()
    
    # Test WandB connection
    wandb_available = wandb_logger.test_connection()
    if wandb_available:
        wandb_logger.init_run(config, run_name=f"housing_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        logging.warning("WandB not available. Continuing without logging to WandB.")
    
    try:
        # Step 1: Load data
        logging.info("Step 1: Loading data")
        X, y = load_california_housing()
        
        # Log dataset info
        if wandb_available:
            wandb_logger.log_metrics({
                "dataset_samples": len(X),
                "dataset_features": len(X.columns)
            })
        
        # Step 2: Preprocess data
        logging.info("Step 2: Preprocessing data")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, 
                                                                   test_size=config["test_size"],
                                                                   random_state=config["random_state"])
        
        # Log preprocessing info
        if wandb_available:
            wandb_logger.log_metrics({
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            })
        
        # Step 3: Train model
        logging.info("Step 3: Training model")
        model = MLModel(config["model_type"])
        model.train(X_train, y_train)
        
        # Step 4: Evaluate model
        logging.info("Step 4: Evaluating model")
        metrics = model.evaluate(X_test, y_test)
        
        # Log metrics to WandB
        if wandb_available:
            wandb_logger.log_metrics(metrics)
            
            # Log performance plots
            y_pred = model.predict(X_test)
            wandb_logger.log_model_performance_plot(y_test, y_pred)
            
            # Log feature importance if available
            if hasattr(model.model, 'feature_importances_'):
                wandb_logger.log_feature_importance(X.columns.tolist(), model.model.feature_importances_)
        
        # Step 5: Save model
        logging.info("Step 5: Saving model")
        os.makedirs("models", exist_ok=True)
        model_path = f"models/housing_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(model_path)
        
        # Print results
        print("\n" + "="*50)
        print("ML PIPELINE RESULTS")
        print("="*50)
        print(f"Dataset: California Housing")
        print(f"Model: {config['model_type']}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        print(f"\nModel saved to: {model_path}")
        print("="*50)
        
        logging.info("ML pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in ML pipeline: {e}")
        raise
    
    finally:
        # Finish WandB run
        if wandb_available:
            wandb_logger.finish_run()

if __name__ == "__main__":
    main()