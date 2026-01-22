"""
main.py

Main script to execute the complete temperature prediction workflow with multiple linear regression.
Each step is documented for learning purposes.
"""
import os
from src.data_loader import load_datasets
from src.linear_regression_model import linear_regression_experiments, linear_regression_rolling_window_experiment

if __name__ == "__main__":
    # Create directory for figures if it doesn't exist
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Load datasets
    train_df, val_df, test_df = load_datasets("data")

    # Run linear regression experiments
    linear_regression_experiments(train_df, test_df, figures_dir=figures_dir)
    
    # Run rolling window experiment
    linear_regression_rolling_window_experiment(train_df, test_df, figures_dir=figures_dir)
