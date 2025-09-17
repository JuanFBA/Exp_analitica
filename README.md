# Exp_analitica

An experimental analytics repository with an automated ML pipeline for the California housing dataset.

## Overview

This repository contains a complete machine learning pipeline that:
- Loads the California housing dataset from scikit-learn
- Preprocesses the data (train/test split, feature scaling)
- Trains a Random Forest regression model
- Evaluates model performance
- Logs metrics and visualizations to Weights & Biases (WandB)
- Saves trained model artifacts

## Features

- **Automated CI/CD**: GitHub Actions workflow that builds, tests, and runs the ML pipeline
- **WandB Integration**: Comprehensive experiment tracking and visualization
- **Robust Data Loading**: Falls back to synthetic data if original dataset is unavailable
- **Model Evaluation**: Comprehensive metrics (RMSE, MAE, R²) and visualizations
- **Artifact Management**: Automatic saving and uploading of trained models

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml     # GitHub Actions workflow
├── src/
│   ├── data_loader.py          # Data loading with fallback
│   ├── preprocessing.py        # Data preprocessing
│   ├── model.py               # ML model training and evaluation
│   └── wandb_logger.py        # WandB logging functionality
├── main.py                    # Main pipeline orchestration
├── requirements.txt           # Python dependencies
└── .gitignore                # Git ignore rules
```

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the ML pipeline**:
   ```bash
   python main.py
   ```

3. **Run individual components**:
   ```bash
   # Test data loading
   python src/data_loader.py
   
   # Test preprocessing
   python src/preprocessing.py
   
   # Test model training
   python src/model.py
   
   # Test WandB logging
   WANDB_MODE=disabled python src/wandb_logger.py
   ```

### GitHub Actions

The pipeline automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual workflow dispatch

To enable WandB logging in GitHub Actions, add your `WANDB_API_KEY` as a repository secret.

## Pipeline Results

The pipeline trains a Random Forest model with excellent performance:
- **R² Score**: ~0.90
- **RMSE**: ~0.36
- **MAE**: ~0.21

Results are logged to WandB with visualizations including:
- Prediction vs actual scatter plots
- Residual analysis
- Feature importance plots
- Model performance metrics

## Dependencies

- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- wandb >= 0.15.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy (for statistical plots)

## Configuration

The pipeline can be configured by modifying the `config` dictionary in `main.py`:

```python
config = {
    "dataset": "california_housing",
    "model_type": "random_forest",  # or "linear_regression"
    "test_size": 0.2,
    "random_state": 42
}
```

## License

This project is for experimental purposes and learning.