"""
Weather Temperature Inference Script

Autoregressive forecasting for arbitrary future dates.
The model predicts TG (mean temperature) autoregressively by:
1. Starting from the most recent 15-day window of actual data
2. Predicting the next 3 days
3. Using those predictions as input for the next forecast step
4. Repeating until the target date is reached

Note: Prediction accuracy degrades with forecast horizon as errors compound.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class Encoder(nn.Module):
    """Encoder with GRU that returns both outputs and hidden states"""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        outputs, hidden = self.gru(x)
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau-style attention mechanism"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):
    """Autoregressive decoder with attention"""
    def __init__(self, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(
            output_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, encoder_outputs, encoder_hidden, forecast_days):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(encoder_outputs.device)
        decoder_hidden = encoder_hidden
        outputs = []

        for t in range(forecast_days):
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)
            gru_input = torch.cat((decoder_input, context.unsqueeze(1)), dim=2)
            gru_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
            prediction = self.fc(gru_output)
            outputs.append(prediction)
            decoder_input = prediction

        return torch.cat(outputs, dim=1)


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model with encoder, decoder, and attention"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, forecast_days=3):
        encoder_outputs, encoder_hidden = self.encoder(x)
        predictions = self.decoder(encoder_outputs, encoder_hidden, forecast_days)
        return predictions


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def load_model_and_scalers(model_path='./best_model_temperature.pth',
                           train_data_path='./modifiedData/train_data.csv'):
    """Load trained model and recreate scalers from training data"""

    print(f"{Colors.CYAN}Loading model and data...{Colors.ENDC}")

    # Load training data to recreate scalers
    train_df = pd.read_csv(train_data_path, index_col='DATE', parse_dates=True)

    # Feature columns (must match training)
    feature_cols = ['TN', 'TX', 'RR', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD']
    target_col = 'TG'

    # Create scalers
    x_scaler = StandardScaler()
    x_scaler.fit(train_df[feature_cols].values)

    y_scaler = StandardScaler()
    y_scaler.fit(train_df[[target_col]].values)

    # Model parameters (must match training)
    n_features = len(feature_cols)
    hidden_dim = 64
    num_layers = 1
    dropout = 0.2

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build model
    encoder = Encoder(n_features, hidden_dim, num_layers, dropout).to(device)
    decoder = Decoder(hidden_dim, 1, num_layers, dropout).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"{Colors.GREEN}Model loaded successfully{Colors.ENDC}")
    print(f"  Device: {device}")
    print(f"  Features: {feature_cols}")

    return model, x_scaler, y_scaler, feature_cols, device


def get_latest_data(data_path='./modifiedData/test_data.csv', n_days=15):
    """Get the most recent n_days of data"""

    # Try to load test data first (most recent)
    df = pd.read_csv(data_path, index_col='DATE', parse_dates=True)

    # Get last n_days
    latest_data = df.iloc[-n_days:]
    last_date = latest_data.index[-1]

    return latest_data, last_date


def compute_historical_stats(train_data_path='./modifiedData/train_data.csv',
                            val_data_path='./modifiedData/val_data.csv',
                            test_data_path='./modifiedData/test_data.csv'):
    """Compute climatological statistics for feature predictions"""

    # Load all data
    train_df = pd.read_csv(train_data_path, index_col='DATE', parse_dates=True)
    val_df = pd.read_csv(val_data_path, index_col='DATE', parse_dates=True)
    test_df = pd.read_csv(test_data_path, index_col='DATE', parse_dates=True)

    # Combine all data
    full_df = pd.concat([train_df, val_df, test_df])

    # Add day-of-year column
    full_df['day_of_year'] = full_df.index.dayofyear

    # Calculate climatology: average for each day of year
    feature_cols = ['TN', 'TX', 'RR', 'SS', 'HU', 'FG', 'FX', 'CC', 'SD']
    climatology = full_df.groupby('day_of_year')[feature_cols].agg(['mean', 'std']).fillna(0)

    # Also calculate TG-TN and TX-TG offsets for reference
    climatology[('TN_offset', 'mean')] = full_df.groupby('day_of_year').apply(
        lambda x: (x['TG'] - x['TN']).mean()
    )
    climatology[('TX_offset', 'mean')] = full_df.groupby('day_of_year').apply(
        lambda x: (x['TX'] - x['TG']).mean()
    )

    stats = {
        'climatology': climatology,
        'feature_cols': feature_cols
    }

    return stats, full_df


def predict_non_tg_features(predicted_tg, forecast_date, historical_stats, add_noise=False):
    """
    Climatology-based feature prediction with optional randomness

    - TG: From model prediction (not used in output, just for TN/TX calculation)
    - TN: TG - day-of-year average(TG - TN)
    - TX: TG + day-of-year average(TX - TG)
    - Others: Day-of-year climatological averages
    - If add_noise=True: Add random variations based on historical std dev
    """

    climatology = historical_stats['climatology']
    feature_cols = historical_stats['feature_cols']

    # Get day of year (handle leap years)
    day_of_year = forecast_date.timetuple().tm_yday

    # Handle Feb 29 (day 60 in leap years) - use Feb 28 climatology
    if day_of_year > 366:
        day_of_year = 365
    if day_of_year == 60 and day_of_year not in climatology.index:
        day_of_year = 59  # Use Feb 28

    # Get climatological values for this day
    if day_of_year in climatology.index:
        clim_day = climatology.loc[day_of_year]
    else:
        # Fallback to nearest day
        nearest_day = min(climatology.index, key=lambda x: abs(x - day_of_year))
        clim_day = climatology.loc[nearest_day]

    # Build feature vector: [TN, TX, RR, SS, HU, FG, FX, CC, SD]
    new_features = np.zeros(9)

    # TN and TX: Use predicted TG with climatological offsets
    tn_offset = clim_day[('TN_offset', 'mean')]
    tx_offset = clim_day[('TX_offset', 'mean')]

    new_features[0] = predicted_tg - tn_offset  # TN
    new_features[1] = predicted_tg + tx_offset  # TX

    # Other features: Use climatological means
    for i, feature in enumerate(feature_cols[2:], start=2):  # Skip TN and TX
        mean_val = clim_day[(feature, 'mean')]
        std_val = clim_day[(feature, 'std')]

        if add_noise and std_val > 0:
            # Add controlled random noise (±1 std dev, 95% of the time)
            noise = np.random.normal(0, std_val * 0.5)  # Use half std dev for gentler variation
            new_features[i] = mean_val + noise
        else:
            new_features[i] = mean_val

    # Add gentle noise to TN/TX if enabled
    if add_noise:
        tn_std = clim_day[('TN', 'std')]
        tx_std = clim_day[('TX', 'std')]
        if tn_std > 0:
            new_features[0] += np.random.normal(0, tn_std * 0.3)
        if tx_std > 0:
            new_features[1] += np.random.normal(0, tx_std * 0.3)

    return new_features


def autoregressive_forecast(model, initial_window, target_date, last_known_date,
                           x_scaler, y_scaler, feature_cols, historical_stats, device, add_noise=False):
    """
    Autoregressively forecast until target_date

    Parameters:
        add_noise: If True, add controlled random variations to non-TG features

    Returns:
        forecast_path: List of (date, temperature) tuples (temperature in 0.1°C units)
        num_steps: Number of autoregressive steps taken
        variance_metrics: Dict with variance and oscillation metrics
    """

    input_days = 15
    forecast_days = 3

    # Initialize
    current_window = initial_window.copy()  # Shape: (15, 9)
    current_date = last_known_date
    forecast_path = []
    all_predictions = []  # Track all predictions for variance calculation

    # Calculate total days to forecast
    days_to_forecast = (target_date - last_known_date).days

    if days_to_forecast <= 0:
        raise ValueError(f"Target date {target_date.date()} must be after last known date {last_known_date.date()}")

    print(f"\n{Colors.CYAN}Starting autoregressive forecast...{Colors.ENDC}")
    print(f"  From: {last_known_date.date()}")
    print(f"  To: {target_date.date()}")
    print(f"  Total days: {days_to_forecast}")

    step_count = 0

    while current_date < target_date:
        step_count += 1

        # Scale input
        window_scaled = x_scaler.transform(current_window)

        # Convert to tensor
        x_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict next 3 days
        with torch.no_grad():
            predictions_scaled = model(x_tensor, forecast_days=forecast_days)

        # Unscale predictions
        predictions = y_scaler.inverse_transform(predictions_scaled.cpu().numpy().reshape(-1, 1)).flatten()

        # Process each prediction and update window
        for i, pred_tg in enumerate(predictions):
            forecast_date = current_date + timedelta(days=i+1)

            if forecast_date <= target_date:
                forecast_path.append((forecast_date, pred_tg))
                all_predictions.append(pred_tg)

            # Update window after each prediction (not just at the end)
            # This allows us to continue forecasting smoothly
            new_features = predict_non_tg_features(pred_tg, forecast_date, historical_stats, add_noise=add_noise)

            # Slide window: remove oldest day, add new prediction
            current_window = np.vstack([current_window[1:], new_features])

        # Update current_date to the last predicted date
        current_date = current_date + timedelta(days=forecast_days)

        if step_count % 10 == 0:
            print(f"  Step {step_count}: Forecasted up to {current_date.date()}")

        # Safety check
        if step_count > 10000:
            raise RuntimeError("Too many iterations - possible infinite loop")

    print(f"{Colors.GREEN}Forecast complete: {step_count} autoregressive steps{Colors.ENDC}")

    # Calculate variance metrics for confidence assessment
    predictions_array = np.array(all_predictions)

    # Convert to actual Celsius for variance calculation
    predictions_celsius = predictions_array / 10.0

    # 1. Overall standard deviation
    overall_std = np.std(predictions_celsius)

    # 2. Oscillation count (direction changes)
    diffs = np.diff(predictions_celsius)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    oscillation_rate = sign_changes / len(diffs) if len(diffs) > 0 else 0

    # 3. Trend stability (compare first half vs second half std)
    if len(predictions_celsius) >= 10:
        mid = len(predictions_celsius) // 2
        first_half_std = np.std(predictions_celsius[:mid])
        second_half_std = np.std(predictions_celsius[mid:])
        stability_ratio = second_half_std / first_half_std if first_half_std > 0 else 1.0
    else:
        stability_ratio = 1.0

    variance_metrics = {
        'overall_std': overall_std,
        'oscillation_rate': oscillation_rate,
        'stability_ratio': stability_ratio,
        'num_predictions': len(predictions_celsius)
    }

    return forecast_path, step_count, variance_metrics


def visualize_forecast(forecast_path, initial_data, target_date, num_steps, save_path='forecast_visualization.png'):
    """Create visualization of the forecast with confidence indicators"""

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot historical data (convert from 0.1°C to actual °C)
    hist_dates = initial_data.index
    hist_temps = initial_data['TG'].values / 10.0  # Convert to actual Celsius
    ax.plot(hist_dates, hist_temps, 'o-', color='blue', linewidth=2,
            label='Historical Data (Last 15 days)', markersize=4)

    # Prepare forecast data (convert from 0.1°C to actual °C)
    forecast_dates = [item[0] for item in forecast_path]
    forecast_temps = [item[1] / 10.0 for item in forecast_path]  # Convert to actual Celsius

    # Color-code by confidence
    # Days 1-3: High confidence (green)
    # Days 4-10: Medium confidence (yellow)
    # Days 10+: Low confidence (red)

    colors = []
    for i in range(len(forecast_dates)):
        if i < 3:
            colors.append('green')
        elif i < 10:
            colors.append('orange')
        else:
            colors.append('red')

    # Plot forecast with color coding
    for i in range(len(forecast_dates)-1):
        ax.plot(forecast_dates[i:i+2], forecast_temps[i:i+2],
               '-', color=colors[i], linewidth=2, alpha=0.7)

    # Plot markers
    ax.scatter(forecast_dates, forecast_temps, c=colors, s=50,
              zorder=5, edgecolors='black', linewidths=0.5)

    # Highlight target date
    target_idx = forecast_dates.index(target_date)
    target_temp = forecast_temps[target_idx]
    ax.scatter([target_date], [target_temp],
              s=200, c='purple', marker='*', zorder=10,
              edgecolors='black', linewidths=1.5,
              label=f'Target Date: {target_temp:.2f}°C')

    # Annotate target date value
    ax.annotate(f'{target_temp:.2f}°C\n{target_date.strftime("%Y-%m-%d")}',
               xy=(target_date, target_temp),
               xytext=(15, 15), textcoords='offset points',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='purple', alpha=0.7, edgecolor='black'),
               color='white',
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                             color='purple', lw=2))

    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'Autoregressive Temperature Forecast ({num_steps} steps)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # Add confidence legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Historical (Known)'),
        Patch(facecolor='green', label='High Confidence (1-3 days)'),
        Patch(facecolor='orange', label='Medium Confidence (4-10 days)'),
        Patch(facecolor='red', label='Low Confidence (10+ days)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n{Colors.GREEN}Visualization saved to: {save_path}{Colors.ENDC}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Weather Temperature Inference')
    parser.add_argument('target_date', type=str,
                       help='Target date for forecast (format: YYYY-MM-DD or DD.MM.YYYY)')
    parser.add_argument('--model', type=str, default='./best_model_temperature.pth',
                       help='Path to trained model file')
    parser.add_argument('--output', type=str, default='forecast_visualization.png',
                       help='Output path for visualization')
    parser.add_argument('--add-noise', action='store_true',
                       help='Add controlled random variations to prevent cycling (improves realism)')

    args = parser.parse_args()

    # Parse target date
    try:
        # Try DD.MM.YYYY format first
        if '.' in args.target_date:
            target_date = datetime.strptime(args.target_date, '%d.%m.%Y')
        else:
            target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
    except ValueError:
        print(f"{Colors.RED}Error: Invalid date format. Use YYYY-MM-DD or DD.MM.YYYY{Colors.ENDC}")
        sys.exit(1)

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}Weather Temperature Autoregressive Forecast{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    # Load model and scalers
    model, x_scaler, y_scaler, feature_cols, device = load_model_and_scalers(args.model)

    # Get latest data
    initial_data, last_known_date = get_latest_data()
    print(f"\n{Colors.CYAN}Latest available data:{Colors.ENDC}")
    print(f"  Last known date: {last_known_date.date()}")
    print(f"  Using last 15 days as starting window")

    # Compute historical statistics for Option C
    historical_stats, full_df = compute_historical_stats()

    # Prepare initial window
    initial_window = initial_data[feature_cols].values

    # Run autoregressive forecast
    if args.add_noise:
        print(f"{Colors.CYAN}Randomness enabled - predictions will vary between runs{Colors.ENDC}")

    try:
        forecast_path, num_steps, variance_metrics = autoregressive_forecast(
            model, initial_window, target_date, last_known_date,
            x_scaler, y_scaler, feature_cols, historical_stats, device, add_noise=args.add_noise
        )
    except ValueError as e:
        print(f"{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)

    # Display results
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}FORECAST RESULTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    target_temp_raw = forecast_path[-1][1]  # Last item in forecast path (in 0.1°C units)
    target_temp = target_temp_raw / 10.0  # Convert to actual Celsius

    print(f"{Colors.BOLD}Target Date:{Colors.ENDC} {target_date.date()}")
    print(f"{Colors.BOLD}Predicted Temperature:{Colors.ENDC} {target_temp:.2f}°C")
    print(f"{Colors.BOLD}Forecast Horizon:{Colors.ENDC} {num_steps} autoregressive steps")

    # Confidence warning
    if num_steps <= 1:
        confidence = "HIGH"
        color = Colors.GREEN
    elif num_steps <= 3:
        confidence = "MEDIUM-HIGH"
        color = Colors.GREEN
    elif num_steps <= 10:
        confidence = "MEDIUM"
        color = Colors.YELLOW
    else:
        confidence = "LOW"
        color = Colors.RED

    print(f"{Colors.BOLD}Confidence Level:{Colors.ENDC} {color}{confidence}{Colors.ENDC}")

    # Display variance metrics
    print(f"\n{Colors.BOLD}Confidence Metrics:{Colors.ENDC}")
    print(f"  Prediction Std Dev: {variance_metrics['overall_std']:.2f}°C")
    print(f"  Oscillation Rate: {variance_metrics['oscillation_rate']:.1%}")
    print(f"  Stability Ratio: {variance_metrics['stability_ratio']:.2f} (later/earlier variance)")

    # Interpret metrics
    if variance_metrics['oscillation_rate'] > 0.6:
        print(f"  {Colors.YELLOW}⚠ High oscillation - predictions are unstable{Colors.ENDC}")
    if variance_metrics['stability_ratio'] > 1.5:
        print(f"  {Colors.YELLOW}⚠ Increasing variance - uncertainty growing{Colors.ENDC}")
    elif variance_metrics['stability_ratio'] < 0.7:
        print(f"  {Colors.GREEN}✓ Decreasing variance - predictions stabilizing{Colors.ENDC}")

    if num_steps > 3:
        print(f"\n{Colors.YELLOW}⚠ Warning: Prediction is {num_steps} steps ahead.{Colors.ENDC}")
        print(f"{Colors.YELLOW}  Accuracy degrades significantly beyond 3-day horizon.{Colors.ENDC}")

    # Show full forecast path (first 10 and last 5) - convert to actual Celsius
    print(f"\n{Colors.BOLD}Forecast Path:{Colors.ENDC}")
    if len(forecast_path) <= 15:
        for date, temp in forecast_path:
            print(f"  {date.date()}: {temp/10:.2f}°C")
    else:
        for date, temp in forecast_path[:10]:
            print(f"  {date.date()}: {temp/10:.2f}°C")
        print(f"  ... ({len(forecast_path) - 15} days omitted) ...")
        for date, temp in forecast_path[-5:]:
            print(f"  {date.date()}: {temp/10:.2f}°C")

    # Create visualization
    visualize_forecast(forecast_path, initial_data, target_date, num_steps, args.output)

    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
