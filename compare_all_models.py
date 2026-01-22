"""
Compare All Weather Prediction Models

Compares the performance of all 3 weather prediction models on Day 1 forecasts:
- GRU (Neural Network)
- Multiple Linear Regression
- Random Forest

Date range: Common overlap starting 2023-01-16
Output: figures/all_models_comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Data file paths
GRU_CSV = 'Neural-Network/gru_predictions.csv'
LR_CSV = 'Linear_Regression/linear_regression_predictions_multi.csv'
RF_CSV = 'ramdom_forest_method/random_forest_predictions.csv'

# Common start date (where all models have predictions)
COMMON_START_DATE = '2023-01-16'

# Output path
OUTPUT_DIR = 'figures'
OUTPUT_FILE = 'all_models_comparison.png'

# Color scheme
COLORS = {
    'actual': 'black',
    'gru': 'blue',
    'lr': 'green',
    'rf': 'orange'
}

print("Loading predictions from all models...")

# Check if GRU predictions exist
if not os.path.exists(GRU_CSV):
    print(f"\nError: GRU predictions file not found at '{GRU_CSV}'")
    print("Please run 'Neural-Network/model_GRU_temperature.py' first to generate GRU predictions.")
    exit(1)

# Load all CSVs
gru_df = pd.read_csv(GRU_CSV)
lr_df = pd.read_csv(LR_CSV)
rf_df = pd.read_csv(RF_CSV)

# Convert date columns to datetime
gru_df['date'] = pd.to_datetime(gru_df['date'])
lr_df['date'] = pd.to_datetime(lr_df['date'])
rf_df['date'] = pd.to_datetime(rf_df['date'])

print(f"  GRU:               {len(gru_df)} samples ({gru_df['date'].min().date()} to {gru_df['date'].max().date()})")
print(f"  Linear Regression: {len(lr_df)} samples ({lr_df['date'].min().date()} to {lr_df['date'].max().date()})")
print(f"  Random Forest:     {len(rf_df)} samples ({rf_df['date'].min().date()} to {rf_df['date'].max().date()})")

print(f"\nAligning data to common date range (from {COMMON_START_DATE})...")

common_start = pd.to_datetime(COMMON_START_DATE)

# Filter each dataframe to common start date
gru_df = gru_df[gru_df['date'] >= common_start].reset_index(drop=True)
lr_df = lr_df[lr_df['date'] >= common_start].reset_index(drop=True)
rf_df = rf_df[rf_df['date'] >= common_start].reset_index(drop=True)

# Find common end date (minimum of all end dates)
common_end = min(gru_df['date'].max(), lr_df['date'].max(), rf_df['date'].max())

# Filter to common end date
gru_df = gru_df[gru_df['date'] <= common_end].reset_index(drop=True)
lr_df = lr_df[lr_df['date'] <= common_end].reset_index(drop=True)
rf_df = rf_df[rf_df['date'] <= common_end].reset_index(drop=True)

print(f"  Common date range: {common_start.date()} to {common_end.date()}")
print(f"  GRU samples:       {len(gru_df)}")
print(f"  LR samples:        {len(lr_df)}")
print(f"  RF samples:        {len(rf_df)}")

# Merge on date to ensure alignment
merged_df = gru_df[['date', 'actual_TG', 'predicted_TG']].rename(
    columns={'predicted_TG': 'gru_pred'}
)
merged_df = merged_df.merge(
    lr_df[['date', 'predicted_TG']].rename(columns={'predicted_TG': 'lr_pred'}),
    on='date', how='inner'
)
merged_df = merged_df.merge(
    rf_df[['date', 'predicted_TG']].rename(columns={'predicted_TG': 'rf_pred'}),
    on='date', how='inner'
)

print(f"  Merged samples:    {len(merged_df)}")

# Extract arrays for calculations
dates = merged_df['date'].values
actual = merged_df['actual_TG'].values
gru_pred = merged_df['gru_pred'].values
lr_pred = merged_df['lr_pred'].values
rf_pred = merged_df['rf_pred'].values

print("\nCalculating metrics...")

def calculate_metrics(actual, predicted):
    """Calculate MAE, RMSE, and R² for predictions."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2

# Calculate metrics for each model
gru_mae, gru_rmse, gru_r2 = calculate_metrics(actual, gru_pred)
lr_mae, lr_rmse, lr_r2 = calculate_metrics(actual, lr_pred)
rf_mae, rf_rmse, rf_r2 = calculate_metrics(actual, rf_pred)

print(f"\n{'Model':<20} {'MAE (0.1°C)':<15} {'MAE (°C)':<12} {'RMSE (0.1°C)':<16} {'RMSE (°C)':<12} {'R²':<10}")
print("-" * 95)
print(f"{'GRU':<20} {gru_mae:<15.2f} {gru_mae/10:<12.2f} {gru_rmse:<16.2f} {gru_rmse/10:<12.2f} {gru_r2:<10.4f}")
print(f"{'Linear Regression':<20} {lr_mae:<15.2f} {lr_mae/10:<12.2f} {lr_rmse:<16.2f} {lr_rmse/10:<12.2f} {lr_r2:<10.4f}")
print(f"{'Random Forest':<20} {rf_mae:<15.2f} {rf_mae/10:<12.2f} {rf_rmse:<16.2f} {rf_rmse/10:<12.2f} {rf_r2:<10.4f}")

print("\nGenerating visualization...")

fig = plt.figure(figsize=(18, 11))

# Create subplot layout: 2 rows
# Row 1: Time series (full width)
# Row 2: 3 scatter plots + 1 bar chart
ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
ax2 = plt.subplot2grid((2, 4), (1, 0))
ax3 = plt.subplot2grid((2, 4), (1, 1))
ax4 = plt.subplot2grid((2, 4), (1, 2))
ax5 = plt.subplot2grid((2, 4), (1, 3))


# Subplot 1: Time Series Comparison
ax1.plot(dates, actual, color=COLORS['actual'], label='Actual', linewidth=1.5, alpha=0.8)
ax1.plot(dates, gru_pred, color=COLORS['gru'], label='GRU', linewidth=1, alpha=0.7)
ax1.plot(dates, lr_pred, color=COLORS['lr'], label='Linear Regression', linewidth=1, alpha=0.7, linestyle='--')
ax1.plot(dates, rf_pred, color=COLORS['rf'], label='Random Forest', linewidth=1, alpha=0.7, linestyle='-.')

ax1.set_title('Day 1 Temperature Forecast - All Models Comparison', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Temperature (0.1°C)', fontsize=10)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Add metrics text box
textstr = (f'GRU:               MAE={gru_mae/10:.2f}°C, RMSE={gru_rmse/10:.2f}°C, R²={gru_r2:.4f}\n'
           f'Linear Regression: MAE={lr_mae/10:.2f}°C, RMSE={lr_rmse/10:.2f}°C, R²={lr_r2:.4f}\n'
           f'Random Forest:     MAE={rf_mae/10:.2f}°C, RMSE={rf_rmse/10:.2f}°C, R²={rf_r2:.4f}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', bbox=props, family='monospace')

# Subplot 2: GRU Scatter Plot
ax2.scatter(actual, gru_pred, color=COLORS['gru'], alpha=0.5, s=10)
min_val, max_val = min(actual.min(), gru_pred.min()), max(actual.max(), gru_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect prediction')
ax2.set_title(f'GRU (R²={gru_r2:.4f})', fontsize=10, fontweight='bold')
ax2.set_xlabel('Actual (0.1°C)', fontsize=9)
ax2.set_ylabel('Predicted (0.1°C)', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

# Subplot 3: Linear Regression Scatter Plot
ax3.scatter(actual, lr_pred, color=COLORS['lr'], alpha=0.5, s=10)
min_val, max_val = min(actual.min(), lr_pred.min()), max(actual.max(), lr_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect prediction')
ax3.set_title(f'Linear Regression (R²={lr_r2:.4f})', fontsize=10, fontweight='bold')
ax3.set_xlabel('Actual (0.1°C)', fontsize=9)
ax3.set_ylabel('Predicted (0.1°C)', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Subplot 4: Random Forest Scatter Plot
ax4.scatter(actual, rf_pred, color=COLORS['rf'], alpha=0.5, s=10)
min_val, max_val = min(actual.min(), rf_pred.min()), max(actual.max(), rf_pred.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect prediction')
ax4.set_title(f'Random Forest (R²={rf_r2:.4f})', fontsize=10, fontweight='bold')
ax4.set_xlabel('Actual (0.1°C)', fontsize=9)
ax4.set_ylabel('Predicted (0.1°C)', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_aspect('equal', adjustable='box')

# Subplot 5: Bar Chart Comparison
models = ['GRU', 'Linear\nRegression', 'Random\nForest']
x = np.arange(len(models))
width = 0.25

# Convert to °C for better readability
mae_values = [gru_mae/10, lr_mae/10, rf_mae/10]
rmse_values = [gru_rmse/10, lr_rmse/10, rf_rmse/10]
r2_values = [gru_r2, lr_r2, rf_r2]

# Create bar chart
bars1 = ax5.bar(x - width, mae_values, width, label='MAE (°C)', color='steelblue')
bars2 = ax5.bar(x, rmse_values, width, label='RMSE (°C)', color='coral')

# Create secondary y-axis for R²
ax5_twin = ax5.twinx()
bars3 = ax5_twin.bar(x + width, r2_values, width, label='R²', color='seagreen')

ax5.set_title('Metrics Comparison', fontsize=10, fontweight='bold')
ax5.set_xlabel('Model', fontsize=9)
ax5.set_ylabel('Error (°C)', fontsize=9)
ax5_twin.set_ylabel('R²', fontsize=9)
ax5.set_xticks(x)
ax5.set_xticklabels(models, fontsize=8)
ax5.set_ylim(0, max(rmse_values) * 1.2)
ax5_twin.set_ylim(0, 1.1)

# Add value labels on bars
for bar, val in zip(bars1, mae_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, rmse_values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars3, r2_values):
    ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Combine legends
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

# Add grid
ax5.grid(True, alpha=0.3, axis='y')

# Final Layout
start_date_str = pd.to_datetime(dates[0]).strftime('%Y-%m-%d')
end_date_str = pd.to_datetime(dates[-1]).strftime('%Y-%m-%d')
plt.suptitle(f'Weather Prediction Model Comparison - Day 1 Forecast ({start_date_str} to {end_date_str})',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save figure
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
plt.savefig(output_path, dpi=800, bbox_inches='tight')
print(f"\nSaved: {output_path}")

plt.show()

# Separate Figure 1: Time Series Comparison (Full Date Range)
print("\nGenerating separate time series graph...")
fig_ts, ax_ts = plt.subplots(figsize=(14, 6))

ax_ts.plot(dates, actual, color=COLORS['actual'], label='Actual', linewidth=1.5, alpha=0.8)
ax_ts.plot(dates, gru_pred, color=COLORS['gru'], label='GRU', linewidth=1, alpha=0.7)
ax_ts.plot(dates, lr_pred, color=COLORS['lr'], label='Linear Regression', linewidth=1, alpha=0.7, linestyle='--')
ax_ts.plot(dates, rf_pred, color=COLORS['rf'], label='Random Forest', linewidth=1, alpha=0.7, linestyle='-.')

ax_ts.set_title('Day 1 Temperature Forecast - All Models Comparison', fontsize=14, fontweight='bold')
ax_ts.set_xlabel('Date', fontsize=11)
ax_ts.set_ylabel('Temperature (0.1°C)', fontsize=11)
ax_ts.legend(loc='upper right', fontsize=10)
ax_ts.grid(True, alpha=0.3)

textstr = (f'GRU:               MAE={gru_mae/10:.2f}°C, RMSE={gru_rmse/10:.2f}°C, R²={gru_r2:.4f}\n'
           f'Linear Regression: MAE={lr_mae/10:.2f}°C, RMSE={lr_rmse/10:.2f}°C, R²={lr_r2:.4f}\n'
           f'Random Forest:     MAE={rf_mae/10:.2f}°C, RMSE={rf_rmse/10:.2f}°C, R²={rf_r2:.4f}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax_ts.text(0.02, 0.98, textstr, transform=ax_ts.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
ts_path = os.path.join(OUTPUT_DIR, 'time_series_comparison.png')
plt.savefig(ts_path, dpi=800, bbox_inches='tight')
print(f"Saved: {ts_path}")
plt.show()

# Separate Figure 2: Time Series Comparison (2025 Only - Zoomed)
print("\nGenerating 2025 zoomed time series graph...")

# Filter for 2025 data
dates_pd = pd.to_datetime(dates)
mask_2025 = dates_pd.year == 2025

if mask_2025.sum() > 0:
    dates_2025 = dates[mask_2025]
    actual_2025 = actual[mask_2025]
    gru_pred_2025 = gru_pred[mask_2025]
    lr_pred_2025 = lr_pred[mask_2025]
    rf_pred_2025 = rf_pred[mask_2025]

    # Calculate metrics for 2025 only
    gru_mae_2025, gru_rmse_2025, gru_r2_2025 = calculate_metrics(actual_2025, gru_pred_2025)
    lr_mae_2025, lr_rmse_2025, lr_r2_2025 = calculate_metrics(actual_2025, lr_pred_2025)
    rf_mae_2025, rf_rmse_2025, rf_r2_2025 = calculate_metrics(actual_2025, rf_pred_2025)

    fig_ts_2025, ax_ts_2025 = plt.subplots(figsize=(14, 6))

    ax_ts_2025.plot(dates_2025, actual_2025, color=COLORS['actual'], label='Actual', linewidth=1.5, alpha=0.8)
    ax_ts_2025.plot(dates_2025, gru_pred_2025, color=COLORS['gru'], label='GRU', linewidth=1, alpha=0.7)
    ax_ts_2025.plot(dates_2025, lr_pred_2025, color=COLORS['lr'], label='Linear Regression', linewidth=1, alpha=0.7, linestyle='--')
    ax_ts_2025.plot(dates_2025, rf_pred_2025, color=COLORS['rf'], label='Random Forest', linewidth=1, alpha=0.7, linestyle='-.')

    ax_ts_2025.set_title('Day 1 Temperature Forecast - 2025 (Detailed View)', fontsize=14, fontweight='bold')
    ax_ts_2025.set_xlabel('Date', fontsize=11)
    ax_ts_2025.set_ylabel('Temperature (0.1°C)', fontsize=11)
    ax_ts_2025.legend(loc='upper right', fontsize=10)
    ax_ts_2025.grid(True, alpha=0.3)

    textstr_2025 = (f'2025 Metrics:\n'
                    f'GRU:               MAE={gru_mae_2025/10:.2f}°C, RMSE={gru_rmse_2025/10:.2f}°C, R²={gru_r2_2025:.4f}\n'
                    f'Linear Regression: MAE={lr_mae_2025/10:.2f}°C, RMSE={lr_rmse_2025/10:.2f}°C, R²={lr_r2_2025:.4f}\n'
                    f'Random Forest:     MAE={rf_mae_2025/10:.2f}°C, RMSE={rf_rmse_2025/10:.2f}°C, R²={rf_r2_2025:.4f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_ts_2025.text(0.02, 0.98, textstr_2025, transform=ax_ts_2025.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    ts_2025_path = os.path.join(OUTPUT_DIR, 'time_series_2025.png')
    plt.savefig(ts_2025_path, dpi=800, bbox_inches='tight')
    print(f"Saved: {ts_2025_path}")
    plt.show()
else:
    print("  No 2025 data available for zoomed view")

# Separate Figure 3: R² Scatter Plots
print("\nGenerating separate R² scatter plots...")
fig_scatter, axes_scatter = plt.subplots(1, 3, figsize=(15, 5))

# GRU Scatter
axes_scatter[0].scatter(actual, gru_pred, color=COLORS['gru'], alpha=0.5, s=15)
min_val, max_val = min(actual.min(), gru_pred.min()), max(actual.max(), gru_pred.max())
axes_scatter[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect prediction')
axes_scatter[0].set_title(f'GRU (R²={gru_r2:.4f})', fontsize=12, fontweight='bold')
axes_scatter[0].set_xlabel('Actual (0.1°C)', fontsize=10)
axes_scatter[0].set_ylabel('Predicted (0.1°C)', fontsize=10)
axes_scatter[0].grid(True, alpha=0.3)
axes_scatter[0].set_aspect('equal', adjustable='box')
axes_scatter[0].legend(fontsize=8)

# Linear Regression Scatter
axes_scatter[1].scatter(actual, lr_pred, color=COLORS['lr'], alpha=0.5, s=15)
min_val, max_val = min(actual.min(), lr_pred.min()), max(actual.max(), lr_pred.max())
axes_scatter[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect prediction')
axes_scatter[1].set_title(f'Linear Regression (R²={lr_r2:.4f})', fontsize=12, fontweight='bold')
axes_scatter[1].set_xlabel('Actual (0.1°C)', fontsize=10)
axes_scatter[1].set_ylabel('Predicted (0.1°C)', fontsize=10)
axes_scatter[1].grid(True, alpha=0.3)
axes_scatter[1].set_aspect('equal', adjustable='box')
axes_scatter[1].legend(fontsize=8)

# Random Forest Scatter
axes_scatter[2].scatter(actual, rf_pred, color=COLORS['rf'], alpha=0.5, s=15)
min_val, max_val = min(actual.min(), rf_pred.min()), max(actual.max(), rf_pred.max())
axes_scatter[2].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect prediction')
axes_scatter[2].set_title(f'Random Forest (R²={rf_r2:.4f})', fontsize=12, fontweight='bold')
axes_scatter[2].set_xlabel('Actual (0.1°C)', fontsize=10)
axes_scatter[2].set_ylabel('Predicted (0.1°C)', fontsize=10)
axes_scatter[2].grid(True, alpha=0.3)
axes_scatter[2].set_aspect('equal', adjustable='box')
axes_scatter[2].legend(fontsize=8)

plt.suptitle('Actual vs Predicted Temperature - Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, 'scatter_plots_comparison.png')
plt.savefig(scatter_path, dpi=800, bbox_inches='tight')
print(f"Saved: {scatter_path}")
plt.show()

# Separate Figure 4: Bar Chart Comparison
print("\nGenerating separate bar chart...")
fig_bar, ax_bar = plt.subplots(figsize=(10, 6))

models = ['GRU', 'Linear Regression', 'Random Forest']
x = np.arange(len(models))
width = 0.25

mae_values = [gru_mae/10, lr_mae/10, rf_mae/10]
rmse_values = [gru_rmse/10, lr_rmse/10, rf_rmse/10]
r2_values = [gru_r2, lr_r2, rf_r2]

bars1 = ax_bar.bar(x - width, mae_values, width, label='MAE (°C)', color='steelblue')
bars2 = ax_bar.bar(x, rmse_values, width, label='RMSE (°C)', color='coral')

ax_bar_twin = ax_bar.twinx()
bars3 = ax_bar_twin.bar(x + width, r2_values, width, label='R²', color='seagreen')

ax_bar.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax_bar.set_xlabel('Model', fontsize=11)
ax_bar.set_ylabel('Error (°C)', fontsize=11)
ax_bar_twin.set_ylabel('R²', fontsize=11)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(models, fontsize=10)
ax_bar.set_ylim(0, max(rmse_values) * 1.3)
ax_bar_twin.set_ylim(0, 1.1)

for bar, val in zip(bars1, mae_values):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, rmse_values):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars3, r2_values):
    ax_bar_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

lines1, labels1 = ax_bar.get_legend_handles_labels()
lines2, labels2 = ax_bar_twin.get_legend_handles_labels()
ax_bar.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

ax_bar.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, 'metrics_bar_chart.png')
plt.savefig(bar_path, dpi=800, bbox_inches='tight')
print(f"Saved: {bar_path}")
plt.show()

print("\nComparison complete!")
