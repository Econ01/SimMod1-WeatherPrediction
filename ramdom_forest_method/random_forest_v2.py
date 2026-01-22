import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Setup ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- 2. Load Datasets ---
train_df = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
test_df = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
# val_df = pd.read_csv('val_data.csv', index_col=0, parse_dates=True)

# --- 3. Define Variables for History ---
ALL_VARS = ['TG', 'TN', 'TX', 'RR', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD']

# --- 4. Core Function: Optimized Feature Engineering (Fixes Fragmentation Warning) ---
def create_lagged_features(df, variables, lag_days=15):
    """
    Creates lagged features efficiently using pd.concat to avoid fragmentation.
    """
    # Initialize a list to hold all columns (Series)
    dfs_to_concat = []
    
    # Add the Target Variable (Today's TG
    target_series = df['TG'].copy()
    target_series.name = 'Target_TG'
    dfs_to_concat.append(target_series)
    
    # Generate Lag Features
    feature_names = []
    
    for var in variables:
        if var not in df.columns:
            continue # Skip if variable is missing
            
        for i in range(1, lag_days + 1):
            col_name = f'{var}_lag_{i}'
            
            # Create the shifted series (data from i days ago)
            shifted_series = df[var].shift(i)
            shifted_series.name = col_name # Naming is crucial for the final DataFrame
            
            # Add to list instead of inserting into DataFrame one by one
            dfs_to_concat.append(shifted_series)
            feature_names.append(col_name)
    
    # Concatenate all columns at once (this fixes the fragmentation warning)
    df_processed = pd.concat(dfs_to_concat, axis=1)
    
    # Clean up: Drop rows with NaNs (the first 'lag_days' rows)
    df_processed.dropna(inplace=True)
    
    return df_processed, feature_names

# --- 5. Apply Transformation ---
print("Generating lagged features (Optimized)...")

train_processed, FEATURE_COLS = create_lagged_features(train_df, ALL_VARS, lag_days=15)
test_processed, _ = create_lagged_features(test_df, ALL_VARS, lag_days=15)

print(f"Total features created: {len(FEATURE_COLS)}") 

# --- 6. Prepare X and y ---
X_train = train_processed[FEATURE_COLS]
y_train = train_processed['Target_TG']

X_test = test_processed[FEATURE_COLS]
y_test = test_processed['Target_TG']

# --- 7. Train Model ---
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# --- 8. Make Predictions ---
print("Making predictions...")
predictions = rf_model.predict(X_test)

# --- 9. Evaluate ---
y_test_celsius = y_test / 10.0
predictions_celsius = predictions / 10.0

mae = mean_absolute_error(y_test_celsius, predictions_celsius)
rmse = np.sqrt(mean_squared_error(y_test_celsius, predictions_celsius))
r2 = r2_score(y_test_celsius, predictions_celsius)

print(f"\nResults on Test Set:")
print(f"MAE:  {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")
print(f"R²:   {r2:.4f}")




# Create a dataframe with dates, actual values, and predictions
results_df = pd.DataFrame({
'date': y_test.index,
'actual_TG': y_test.values, # Keep in original units (0.1°C)
'predicted_TG': predictions # Keep in original units (0.1°C)
})
# Save to CSV
results_df.to_csv('random_forest_predictions.csv', index=False)
print("Predictions saved to 'random_forest_predictions.csv'")

# Feature Importance
print("\nTop 5 Most Important Features:")
feature_importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
print(feature_importances.sort_values(ascending=False).head(5))

