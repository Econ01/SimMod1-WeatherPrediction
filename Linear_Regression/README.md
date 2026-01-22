# Temperature Prediction with Linear Regression

This project is a learning guide for predicting daily mean temperature (TG) using multiple linear regression and historical weather data from the European Climate Assessment & Dataset (ECA&D).

## Project Structure

- `src/data_loader.py`: Loading and cleaning of relevant weather data.
- `src/linear_regression_model.py`: Training, evaluation, and visualization of the linear regression model.
- `main.py`: Main script to execute the entire workflow.
- `data/`: Folder containing the original data files.
- `figures/`: Folder for generated plots.
- `notebooks/`: For experiments and exploratory analysis.

## Variables Used

- **TG**: Mean daily temperature (Target)
- **TN**: Minimum daily temperature
- **TX**: Maximum daily temperature

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- matplotlib

Install dependencies with:
```bash
pip install pandas scikit-learn matplotlib
```

## Methodology & Experiments

The project implements two experimental approaches to linear regression to demonstrate different concepts.

### 1. Simple Linear Regression (Forecasting)
*   **Goal**: Predict today's mean temperature (`TG`) using *only* yesterday's mean temperature (`TG_lag1`).
*   **Type**: Time series forecasting.
*   **Validity**: This is a valid predictive approach as it uses past data to predict the future.

### 2. Multiple Linear Regression (Forecasting with Multiple Features)
*   **Goal**: Predict today's mean temperature (`TG`) using *yesterday's* minimum (`TN_lag1`), maximum (`TX_lag1`), and mean (`TG_lag1`) temperatures.
*   **Type**: Time series forecasting with multiple predictors.
*   **Validity**: This is a valid predictive approach as it uses historical data from the previous day to predict today's temperature, avoiding data leakage.

### 3. Rolling Window Linear Regression (Feature Engineering)
*   **Goal**: Predict today's mean temperature (`TG`) using yesterday's data (`TN_lag1`, `TX_lag1`, `TG_lag1`) plus the **average mean temperature of the last 3 days** (`TG_mean_3d`).
*   **Type**: Time series forecasting with rolling statistics.
*   **Validity**: Uses a shifted window to ensure only past data (from t-1, t-2, and t-3) is used for the current prediction.

## Results

The following table summarizes the performance of the models on the test set (2023-2025):

| Experiment | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Simple Regression** (TG_lag1 → TG) | 17.606 | 22.607 | 0.880 |
| **Multiple Regression** (TN_lag1, TX_lag1, TG_lag1 → TG) | 17.205 | 22.204 | 0.885 |
| **Rolling Window Regression** (Multi + TG_mean_3d) | 17.181 | 22.235 | 0.884 |

> **Note**: Both models now use valid forecasting approaches with lagged features, avoiding data leakage.

### Visualizations

#### Simple Linear Regression: Real vs Predicted
![Simple Regression Results](https://raw.githubusercontent.com/DiegoGarces95/Linear_Regression_SimMod1-WeatherPrediction/main/figures/tg_simple_real_vs_predicho.png)

#### Multiple Linear Regression: Real vs Predicted
![Multiple Regression Results](https://raw.githubusercontent.com/DiegoGarces95/Linear_Regression_SimMod1-WeatherPrediction/main/figures/tg_multi_real_vs_predicho.png)

#### Rolling Window Regression: Real vs Predicted
![Rolling Window Results](https://raw.githubusercontent.com/DiegoGarces95/Linear_Regression_SimMod1-WeatherPrediction/main/figures/tg_rolling_real_vs_predicho.png)

## Execution

Run `main.py` to load data, train models, and generate results and visualizations:

```bash
python main.py
```
