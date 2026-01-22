"""
linear_regression_model.py
Script for training, evaluation, and visualization of a multiple linear regression model.
Includes detailed comments for learning.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def train_and_evaluate(data, test_size=0.2, random_state=42):
    """
    Trains a multiple linear regression model to predict mean temperature (TG).
    Splits data into training and testing sets, trains the model, and evaluates its performance.
    Displays metrics and a plot of actual vs. predicted results.
    """
    # 1. Select predictor variables and target
    X = data.drop(columns=['DATE', 'TG'])
    y = data['TG']

    # 2. Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 3. Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Make predictions
    y_pred = model.predict(X_test)

    # 5. Evaluate the model with standard metrics
    mae = mean_absolute_error(y_test, y_pred)
    # Calcluate RMSE manually for compatibility
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    # 6. Result visualization
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted Temperature (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.show()

    return model

def linear_regression_experiments(train_df, test_df, figures_dir="figures"):
    """
    Performs two experiments:
    1. Simple Linear Regression: TG_lag1 -> TG
    2. Multiple Linear Regression: TN, TX, TG -> TG
    Saves results and plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    # --- Simple Linear Regression ---
    # Combine train and test to create proper lag features
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create lag feature on combined data
    combined_df["TG_lag1"] = combined_df["TG"].shift(1)
    
    # Now split back into train and test
    n_train = len(train_df)
    train_simple = combined_df[:n_train].dropna() # Remove first row of train (no lag)
    test_simple = combined_df[n_train:].copy()   # Test keeps all rows (has lag from last train value)

    X_train_simple = train_simple[["TG_lag1"]]
    y_train_simple = train_simple["TG"]
    X_test_simple = test_simple[["TG_lag1"]]
    y_test_simple = test_simple["TG"]
    
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train_simple)
    pred_simple = model_simple.predict(X_test_simple)
    
    mae_simple = mean_absolute_error(y_test_simple, pred_simple)
    rmse_simple = mean_squared_error(y_test_simple, pred_simple) ** 0.5
    r2_simple = r2_score(y_test_simple, pred_simple)
    
    results_simple = pd.DataFrame({
        "date": test_simple["DATE"],
        "actual_TG": y_test_simple,
        "predicted_TG": pred_simple
    })
    results_simple.to_csv("linear_regression_predictions_simple.csv", index=False)
    plt.figure(figsize=(10,4))
    plt.plot(y_test_simple.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred_simple, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Simple Linear Regression: Actual vs Predicted TG')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Temperature (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_simple_real_vs_predicho.png'))
    plt.close()
    print("--- Simple Linear Regression ---")
    print(f"MAE: {mae_simple:.3f}")
    print(f"RMSE: {rmse_simple:.3f}")
    print(f"R²: {r2_simple:.3f}")
    print("Results saved in linear_regression_predictions_simple.csv and figures/tg_simple_real_vs_predicho.png\n")
    # --- Multiple Linear Regression (CORRECTED) ---
    # Combine train and test to create proper lag features across datasets
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create lagged features from yesterday's weather
    combined_df["TN_lag1"] = combined_df["TN"].shift(1)
    combined_df["TX_lag1"] = combined_df["TX"].shift(1)
    combined_df["TG_lag1"] = combined_df["TG"].shift(1)
    
    # Split back into train and test
    n_train = len(train_df)
    train_multi = combined_df[:n_train].dropna()  # Remove first row (no lag available)
    test_multi = combined_df[n_train:].copy()  # Keep all test rows (has lag from last train value)
    
    # Prepare features and targets
    feature_cols = ["TN_lag1", "TX_lag1", "TG_lag1"]
    X_train_multi = train_multi[feature_cols]
    y_train_multi = train_multi["TG"]
    X_test_multi = test_multi[feature_cols]
    y_test_multi = test_multi["TG"]
    
    # Train model
    model_multi = LinearRegression()
    model_multi.fit(X_train_multi, y_train_multi)
    
    # Make predictions
    pred_multi = model_multi.predict(X_test_multi)
    
    # Calculate metrics
    mae_multi = mean_absolute_error(y_test_multi, pred_multi)
    rmse_multi = mean_squared_error(y_test_multi, pred_multi) ** 0.5
    r2_multi = r2_score(y_test_multi, pred_multi)
    
    # Save results
    results_multi = pd.DataFrame({
        "date": test_multi["DATE"],
        "actual_TG": y_test_multi,
        "predicted_TG": pred_multi
    })
    results_multi.to_csv("linear_regression_predictions_multi.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10,4))
    plt.plot(y_test_multi.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred_multi, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Multiple Linear Regression: Actual vs Predicted TG')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Temperature (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_multi_real_vs_predicho.png'))
    plt.close()
    
    print("--- Multiple Linear Regression ---")
    print(f"MAE: {mae_multi:.3f}")
    print(f"RMSE: {rmse_multi:.3f}")
    print(f"R²: {r2_multi:.3f}")
    print("Results saved in linear_regression_predictions_multi.csv and figures/tg_multi_real_vs_predicho.png\n")

def linear_regression_rolling_window_experiment(train_df, test_df, figures_dir="figures"):
    """
    Performs an experiment using a rolling window mean of TG as a feature.
    Calculates the mean of TG over the last 3 days (t-1, t-2, t-3).
    Features: TN_lag1, TX_lag1, TG_lag1, TG_mean_3d
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Combined df for lag/rolling calculation
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Standard lags
    combined_df["TN_lag1"] = combined_df["TN"].shift(1)
    combined_df["TX_lag1"] = combined_df["TX"].shift(1)
    combined_df["TG_lag1"] = combined_df["TG"].shift(1)
    
    # Rolling mean of TG (window=3, shifted by 1 to use past data only)
    # .shift(1) makes the window [t-1, t-2, t-3] relative to current t
    combined_df["TG_mean_3d"] = combined_df["TG"].shift(1).rolling(window=3).mean()
    
    # Split back
    n_train = len(train_df)
    # We need to drop more rows now because of the 3-day window
    # First 3 rows will have NaN for TG_mean_3d
    train_rolling = combined_df[:n_train].dropna()
    test_rolling = combined_df[n_train:].copy()
    
    # Prepare features
    feature_cols = ["TN_lag1", "TX_lag1", "TG_lag1", "TG_mean_3d"]
    X_train = train_rolling[feature_cols]
    y_train = train_rolling["TG"]
    X_test = test_rolling[feature_cols]
    y_test = test_rolling["TG"]
    
    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    pred = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    
    # Save results
    results = pd.DataFrame({
        "date": test_rolling["DATE"],
        "actual_TG": y_test,
        "predicted_TG": pred
    })
    results.to_csv("linear_regression_predictions_rolling.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10,4))
    plt.plot(y_test.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(pred, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Rolling Window Regression: Actual vs Predicted TG')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Temperature (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tg_rolling_real_vs_predicho.png'))
    plt.close()
    
    print("--- Rolling Window Regression ---")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print("Results saved in linear_regression_predictions_rolling.csv and figures/tg_rolling_real_vs_predicho.png\n")
    return model
