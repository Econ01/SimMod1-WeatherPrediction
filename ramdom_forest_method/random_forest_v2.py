import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Setup ---
# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- 2. Load Datasets ---

train_df = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
test_df = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)
# val_df = pd.read_csv('val_data.csv', index_col=0, parse_dates=True)


# --- 3. Define Variables for History ---
# We want to use the history (lag) of ALL variables to predict today's temperature.
# Note: 'TG' is added here because now we're using past 15 days' features to predict today's TG
ALL_VARS = ['TG', 'TN', 'TX', 'RR', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD']

# --- 4. Core Function: Create 15-Day Lagged Features ---
def create_lagged_features(df, variables, lag_days=15):
    """
    Creates a new DataFrame where each row contains the target (Today's TG)
    and features from the past 'lag_days' for all specified variables.
    """
    df_processed = pd.DataFrame(index=df.index)
    
    # A. Set the Target: Today's Mean Temperature (TG)
    df_processed['Target_TG'] = df['TG']
    
    # B. Generate Features: Loop through all variables and all lag days
    feature_names = []
    
    for var in variables:
        if var not in df.columns:
            continue # Skip if the variable doesn't exist in the data
            
        for i in range(1, lag_days + 1):
            col_name = f'{var}_lag_{i}'
            # shift(i) moves data down by i rows (gets data from i days ago)
            # Crucial: This ensures we strictly use PAST data to predict the PRESENT.
            df_processed[col_name] = df[var].shift(i)
            feature_names.append(col_name)
    
    # C. Clean up: Drop the first few rows (NaNs) because they don't have enough history
    df_processed.dropna(inplace=True)
    
    return df_processed, feature_names

# --- 5. Apply Transformation ---
print("Generating lagged features (this might take a moment)...")

# Process Training Data
train_processed, FEATURE_COLS = create_lagged_features(train_df, ALL_VARS, lag_days=15)

# Process Test Data
test_processed, _ = create_lagged_features(test_df, ALL_VARS, lag_days=15)

print(f"Total features created: {len(FEATURE_COLS)}") 
# Example: 10 variables * 15 days = 150 features

# --- 6. Prepare X and y ---
X_train = train_processed[FEATURE_COLS]
y_train = train_processed['Target_TG']

X_test = test_processed[FEATURE_COLS]
y_test = test_processed['Target_TG']

# --- 7. Train Model ---
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    random_state=SEED, # Ensure consistent results
    n_jobs=-1          # Use all CPU cores for speed
)
rf_model.fit(X_train, y_train)

# --- 8. Make Predictions ---
print("Making predictions...")
predictions = rf_model.predict(X_test)

# --- 9. Evaluate (Convert Units) ---
# ECAD data is stored in 0.1°C units (e.g., 123 = 12.3°C).
y_test_celsius = y_test / 10.0
predictions_celsius = predictions / 10.0

mae = mean_absolute_error(y_test_celsius, predictions_celsius)
rmse = np.sqrt(mean_squared_error(y_test_celsius, predictions_celsius))
r2 = r2_score(y_test_celsius, predictions_celsius)

print(f"\nResults on Test Set:")
print(f"MAE:  {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")
print(f"R²:   {r2:.4f}")

# (Optional) Feature Importance Analysis
feature_importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
print("\nTop 5 Most Important Features:")
print(feature_importances.sort_values(ascending=False).head(5))
