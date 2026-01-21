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
val_df = pd.read_csv('val_data.csv', index_col=0, parse_dates=True)

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
val_processed, _              = create_lagged_features(val_df,   ALL_VARS, lag_days=15)
test_processed, _ = create_lagged_features(test_df, ALL_VARS, lag_days=15)

print(f"Total features created: {len(FEATURE_COLS)}") 





# --- 6. Prepare X and y ---
X_train = train_processed[FEATURE_COLS]
y_train = train_processed['Target_TG']

X_val   = val_processed[FEATURE_COLS]  
y_val   = val_processed['Target_TG']    

X_test = test_processed[FEATURE_COLS]
y_test = test_processed['Target_TG']



from sklearn.feature_selection import SelectFromModel

# --- 6.5 Feature Selection (Model-Based) ---
print("Starting Feature Selection...")

# Initialize a temporary Random Forest to evaluate feature importance
# We use standard parameters here; this model is just for screening, not prediction.
selector_model = RandomForestRegressor(
    n_estimators=100,
    random_state=SEED,
    n_jobs=-1
)

# Fit the selector on the Training set
print(f"   Fitting selector model on {X_train.shape[1]} features...")
selector_model.fit(X_train, y_train)

# Configure the Selector
# Strategy: Force the model to keep only the Top 30 most important features.
# This reduces dimensionality (150 -> 30) and removes noise/redundancy.
selector = SelectFromModel(
    selector_model,
    max_features=30,  # Strict limit: keep top 30
    threshold=-np.inf, # Required when using max_features
    prefit=True        # We already fitted the model above
)

# Transform (Filter) the datasets
# IMPORTANT: We must apply the same filter to Train, Validation, and Test sets.
X_train_selected = selector.transform(X_train)
X_val_selected   = selector.transform(X_val)
X_test_selected  = selector.transform(X_test)

# Update the list of Feature Names (for plotting later)
selected_indices = selector.get_support(indices=True)
SELECTED_COLS = [FEATURE_COLS[i] for i in selected_indices]

# --- Sanity Check / Report ---
print(f"\n✅ Feature Selection Complete.")
print(f"   Original Feature Count: {X_train.shape[1]}")
print(f"   Selected Feature Count: {X_train_selected.shape[1]}")
print(f"   Top 5 Retained Features: {SELECTED_COLS[:5]}")

# ⚠️ 
# must now use:
# X_train_selected, X_val_selected, and X_test_selected
# instead of the original X_train, X_val, X_test.




# --- 7. Hyperparameter Tuning (Updated) ---
# Previous approach: Trained a single model with fixed parameters
# Updated: Loop through parameter combinations to select the best model based on validation Set performance

print("Starting Hyperparameter Tuning on Validation Set...")

# Define the parameter grid for testing (Grid Search)
param_grid = [
    {'n_estimators': 100, 'max_depth': None}, # 1: default
    {'n_estimators': 100, 'max_depth': 10},   # 2: Limit depth (prevent overfitting)
    {'n_estimators': 200, 'max_depth': None}, # 3: increased num of trees
    {'n_estimators': 50,  'max_depth': 10}    # 4: lightweight model(faster training)
]

best_score = float('inf') # Initialize best score as infinity
best_model = None         # Placeholder to store the best performing model

for params in param_grid:
    print(f"Testing params: {params} ...", end=" ")

    # Initialize temporary model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=SEED,
        n_jobs=-1
    )

    # Train on training set
    model.fit(X_train_selected, y_train)

    # Critical: evaluate on validation set
    val_preds = model.predict(X_val_selected)
    val_rmse = np.sqrt(mean_squared_error(y_val/10.0, val_preds/10.0)) # to celsius

    print(f"Val RMSE: {val_rmse:.4f}")

    # Model selection: keeep the model that performs better
    if val_rmse < best_score:
        best_score = val_rmse
        best_model = model # Store the trained model from this round

print("-" * 30)
print(f" Best Validation RMSE: {best_score:.4f}")
print(f" Model used for Final Test: {best_model.get_params()['n_estimators']} trees, Depth: {best_model.get_params()['max_depth']}")





# --- 8. Make Predictions ---
#print("Making predictions...")
#predictions = rf_model.predict(X_test)

print("\nMaking predictions using the BEST model...")

predictions = best_model.predict(X_test_selected)




# --- 9. Evaluate ---
y_test_celsius = y_test / 10.0
predictions_celsius = predictions / 10.0

mae = mean_absolute_error(y_test_celsius, predictions_celsius)
rmse = np.sqrt(mean_squared_error(y_test_celsius, predictions_celsius))
r2 = r2_score(y_test_celsius, predictions_celsius)

print(f"\n✅ Final Results on Test Set:")
print(f"MAE:  {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")
print(f"R²:   {r2:.4f}")




# Feature Importance
print("\nTop 5 Most Important Features:")
feature_importances = pd.Series(best_model.feature_importances_, index=SELECTED_COLS)
print(feature_importances.sort_values(ascending=False).head(5))







