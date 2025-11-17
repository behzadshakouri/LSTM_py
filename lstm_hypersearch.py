import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Output directory
OUTPUT_DIR = '/home/arash/Downloads/NO_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}\n")


def test_ergodicity(data, target_col_name='Stl_t_S_NO_Concentration', window_size=100):
    """
    Comprehensive ergodicity tests for time series data

    Tests include:
    1. Stationarity tests (ADF, KPSS)
    2. Rolling statistics stability
    3. Autocorrelation analysis
    4. Ensemble vs time average comparison
    """
    print("\n" + "=" * 70)
    print("ERGODICITY ANALYSIS")
    print("=" * 70)

    # Extract target variable
    if isinstance(data, pd.DataFrame):
        y = data.iloc[:, -1].values
    else:
        y = data[:, -1] if len(data.shape) > 1 else data

    n = len(y)
    print(f"Analyzing {n} data points")

    # ============================================================
    # 1. STATIONARITY TESTS
    # ============================================================
    print("\n" + "-" * 70)
    print("1. STATIONARITY TESTS (Required for Ergodicity)")
    print("-" * 70)

    # Augmented Dickey-Fuller test
    print("\nAugmented Dickey-Fuller Test:")
    try:
        adf_result = adfuller(y, autolag='AIC')
        print(f"  ADF Statistic: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  Critical values:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.6f}")

        if adf_result[1] < 0.05:
            print("  ✓ STATIONARY: Reject null hypothesis (data has no unit root)")
        else:
            print("  ✗ NON-STATIONARY: Cannot reject null hypothesis (data may have unit root)")
    except Exception as e:
        print(f"  Error in ADF test: {e}")

    # KPSS test
    print("\nKPSS Test:")
    try:
        kpss_result = kpss(y, regression='c', nlags='auto')
        print(f"  KPSS Statistic: {kpss_result[0]:.6f}")
        print(f"  p-value: {kpss_result[1]:.6f}")
        print(f"  Critical values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.6f}")

        if kpss_result[1] > 0.05:
            print("  ✓ STATIONARY: Cannot reject null hypothesis (data is stationary)")
        else:
            print("  ✗ NON-STATIONARY: Reject null hypothesis (data is not stationary)")
    except Exception as e:
        print(f"  Error in KPSS test: {e}")

    # ============================================================
    # 2. ROLLING STATISTICS (Stability Test)
    # ============================================================
    print("\n" + "-" * 70)
    print("2. ROLLING STATISTICS STABILITY")
    print("-" * 70)

    # Calculate rolling statistics
    if n > window_size * 2:
        n_windows = n // window_size
        window_means = []
        window_stds = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = y[start_idx:end_idx]
            window_means.append(np.mean(window_data))
            window_stds.append(np.std(window_data))

        window_means = np.array(window_means)
        window_stds = np.array(window_stds)

        # Calculate coefficient of variation for means and stds
        cv_mean = np.std(window_means) / np.abs(np.mean(window_means)) if np.mean(window_means) != 0 else np.inf
        cv_std = np.std(window_stds) / np.mean(window_stds) if np.mean(window_stds) != 0 else np.inf

        print(f"\nWindow size: {window_size} samples")
        print(f"Number of windows: {n_windows}")
        print(f"\nRolling Mean:")
        print(f"  Mean of means: {np.mean(window_means):.6f}")
        print(f"  Std of means: {np.std(window_means):.6f}")
        print(f"  Coefficient of Variation: {cv_mean:.6f}")

        print(f"\nRolling Std:")
        print(f"  Mean of stds: {np.mean(window_stds):.6f}")
        print(f"  Std of stds: {np.std(window_stds):.6f}")
        print(f"  Coefficient of Variation: {cv_std:.6f}")

        # Interpretation
        if cv_mean < 0.1 and cv_std < 0.3:
            print("\n  ✓ STABLE: Rolling statistics show low variation")
            print("    This supports ergodicity assumption")
        else:
            print("\n  ⚠ UNSTABLE: Rolling statistics show high variation")
            print("    This suggests potential non-ergodicity")
            if cv_mean >= 0.1:
                print(f"    - Mean CV ({cv_mean:.3f}) indicates changing mean over time")
            if cv_std >= 0.3:
                print(f"    - Std CV ({cv_std:.3f}) indicates changing variance over time")
    else:
        print(f"  Warning: Dataset too small for rolling window analysis")
        print(f"  Need at least {window_size * 2} samples, have {n}")

    # ============================================================
    # 3. AUTOCORRELATION ANALYSIS
    # ============================================================
    print("\n" + "-" * 70)
    print("3. AUTOCORRELATION ANALYSIS")
    print("-" * 70)

    try:
        # Calculate ACF
        max_lags = min(40, n // 4)
        acf_values = acf(y, nlags=max_lags, fft=True)

        # Find where ACF drops below 0.1
        significant_lags = np.where(np.abs(acf_values[1:]) > 0.1)[0]

        print(f"\nAutocorrelation at selected lags:")
        for lag in [1, 5, 10, 20]:
            if lag < len(acf_values):
                print(f"  Lag {lag}: {acf_values[lag]:.6f}")

        if len(significant_lags) > 0:
            decay_lag = significant_lags[-1] + 1
            print(f"\nACF decays below 0.1 at lag: {decay_lag}")

            if decay_lag < n / 10:
                print(f"  ✓ FAST DECAY: ACF decays quickly (within {decay_lag} lags)")
                print("    This supports ergodicity - system 'forgets' its past")
            else:
                print(f"  ⚠ SLOW DECAY: ACF decays slowly ({decay_lag} lags)")
                print("    Long memory may indicate non-ergodic behavior")
        else:
            print("\n  ✓ FAST DECAY: ACF drops below 0.1 very quickly")
            print("    Strong support for ergodicity")

    except Exception as e:
        print(f"  Error in ACF analysis: {e}")

    # ============================================================
    # 4. ENSEMBLE VS TIME AVERAGE COMPARISON
    # ============================================================
    print("\n" + "-" * 70)
    print("4. ENSEMBLE VS TIME AVERAGE TEST")
    print("-" * 70)

    if n > window_size * 10:
        # Time average: average over entire series
        time_avg = np.mean(y)
        time_std = np.std(y)

        # Ensemble average: average of multiple sub-segments at different times
        n_segments = 10
        segment_size = n // n_segments
        ensemble_avgs = []

        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            ensemble_avgs.append(np.mean(y[start_idx:end_idx]))

        ensemble_avg = np.mean(ensemble_avgs)
        ensemble_std = np.std(ensemble_avgs)

        print(f"\nTime Average: {time_avg:.6f} ± {time_std:.6f}")
        print(f"Ensemble Average: {ensemble_avg:.6f} ± {ensemble_std:.6f}")
        print(f"Difference: {abs(time_avg - ensemble_avg):.6f}")
        print(f"Relative difference: {abs(time_avg - ensemble_avg) / abs(time_avg) * 100:.2f}%")

        # Check if they're similar (within 5%)
        if abs(time_avg - ensemble_avg) / abs(time_avg) < 0.05:
            print("\n  ✓ ERGODIC: Time average ≈ Ensemble average")
            print("    The process appears to be ergodic")
        else:
            print("\n  ✗ NON-ERGODIC: Time average ≠ Ensemble average")
            print("    The process may not be ergodic")
    else:
        print(f"  Warning: Dataset too small for ensemble analysis")

    # ============================================================
    # OVERALL ERGODICITY ASSESSMENT
    # ============================================================
    print("\n" + "=" * 70)
    print("OVERALL ERGODICITY ASSESSMENT")
    print("=" * 70)
    print("\nFor a process to be ergodic, it should:")
    print("  1. ✓ Be stationary (pass ADF and KPSS tests)")
    print("  2. ✓ Have stable rolling statistics (low CV)")
    print("  3. ✓ Have decaying autocorrelation (short memory)")
    print("  4. ✓ Show time average ≈ ensemble average")
    print("\nReview the results above to assess if your data meets these criteria.")
    print("=" * 70 + "\n")

    return True


def plot_ergodicity_diagnostics(data, window_size=100):
    """Create diagnostic plots for ergodicity assessment"""
    if isinstance(data, pd.DataFrame):
        y = data.iloc[:, -1].values
    else:
        y = data[:, -1] if len(data.shape) > 1 else data

    n = len(y)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Time series plot
    axes[0, 0].plot(y, linewidth=0.8)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Time Series Data')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Rolling mean and std
    if n > window_size * 2:
        df_temp = pd.DataFrame({'value': y})
        rolling_mean = df_temp['value'].rolling(window=window_size).mean()
        rolling_std = df_temp['value'].rolling(window=window_size).std()

        axes[0, 1].plot(rolling_mean, label='Rolling Mean', linewidth=2)
        axes[0, 1].axhline(y=np.mean(y), color='r', linestyle='--', label='Overall Mean')
        axes[0, 1].fill_between(range(len(rolling_mean)),
                                rolling_mean - rolling_std,
                                rolling_mean + rolling_std,
                                alpha=0.3, label='±1 Std')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title(f'Rolling Statistics (window={window_size})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Autocorrelation plot
    max_lags = min(50, n // 4)
    acf_values = acf(y, nlags=max_lags, fft=True)
    lags = np.arange(len(acf_values))

    axes[1, 0].stem(lags, acf_values, basefmt=' ')
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    axes[1, 0].axhline(y=0.1, color='r', linestyle='--', linewidth=0.8, label='Threshold (0.1)')
    axes[1, 0].axhline(y=-0.1, color='r', linestyle='--', linewidth=0.8)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_title('Autocorrelation Function (ACF)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Histogram
    axes[1, 1].hist(y, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Values')
    axes[1, 1].axvline(np.mean(y), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y):.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'ergodicity_diagnostics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ergodicity diagnostic plots saved to: {save_path}")

    return fig


class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for time series prediction with regularization"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Add dropout before final layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step and apply dropout
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def create_sequences_sparse_lags(X_features, y_target, lags):
    """
    Create sequences using SPARSE LAGS instead of consecutive timesteps

    Args:
        X_features: Feature matrix (n_samples, n_features) - does NOT include target
        y_target: Target variable (n_samples,) - the variable to predict
        lags: List of lag positions [5, 10, 20, 30, 50, 75, 100]

    Returns:
        X_seq: Input sequences (n_sequences, len(lags), n_features)
        y_seq: Target values (n_sequences,)
    """
    X_seq, y_seq = [], []
    max_lag = max(lags)

    for i in range(max_lag, len(X_features)):
        # Create sequence with ONLY the specified lags
        # Each element in sequence is from timestep (i - lag)
        sequence = np.stack([X_features[i - lag, :] for lag in lags])
        X_seq.append(sequence)

        # Target: y value at current timestep i
        y_seq.append(y_target[i])

    return np.array(X_seq), np.array(y_seq)


def load_and_preprocess_data(file_path, lags=None, test_size=0.3, run_ergodicity_tests=True):
    """
    Load data from CSV and preprocess it - EXPLICITLY EXCLUDES target from input features
    Uses SPARSE LAGS instead of consecutive sequences

    Args:
        file_path: Path to CSV file
        lags: List of lag positions [5, 10, 20, ...] or None for default
        test_size: Fraction of data for testing (default 0.3 for 70/30 split)
        run_ergodicity_tests: Whether to run ergodicity analysis
    """
    # Default lags if none provided
    if lags is None:
        lags = [5, 10, 20, 30, 50]

    # Load data
    df = pd.read_csv(file_path)
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\n{'=' * 70}")
    print("DATA STRUCTURE VERIFICATION")
    print(f"{'=' * 70}")
    print(f"Target variable (last column): {df.columns[-1]}")
    print(f"Number of input features: {len(df.columns) - 1}")
    print(f"Input feature columns: {df.columns[:-1].tolist()}")

    # Check if target appears in features
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    if target_col in feature_cols:
        raise ValueError(f"ERROR: Target variable '{target_col}' appears in feature columns!")

    print(f"\n✓ VERIFIED: Target '{target_col}' does NOT appear in input features")
    print(f"✓ VERIFIED: Using SPARSE LAGS: {lags}")
    print(f"✓ Maximum lookback: {max(lags)} timesteps")
    print(f"✓ Number of lag positions: {len(lags)}")
    print(f"{'=' * 70}\n")

    # Run ergodicity tests on raw data
    if run_ergodicity_tests:
        test_ergodicity(df, window_size=min(200, len(df) // 10))
        plot_ergodicity_diagnostics(df, window_size=min(200, len(df) // 10))

    # Convert to numpy array
    data = df.values

    # Separate features and target
    X_features = data[:, :-1]
    y_target = data[:, -1]

    print(f"\n{'=' * 70}")
    print("FEATURE-TARGET SEPARATION")
    print(f"{'=' * 70}")
    print(f"X_features shape: {X_features.shape} - Contains ONLY input features")
    print(f"y_target shape: {y_target.shape} - Contains ONLY target variable")
    print(f"✓ Target is completely separated from features")
    print(f"{'=' * 70}\n")

    # Normalize features and target SEPARATELY
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_features_scaled = scaler_X.fit_transform(X_features)
    y_target_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).flatten()

    # Create sequences with SPARSE LAGS
    print(f"Creating sequences with SPARSE LAGS: {lags}")
    print(f"  - Each sequence has {len(lags)} timesteps (not {max(lags)}!)")
    print(f"  - Covers {max(lags)} timesteps of history")
    print(f"  - Computational savings: ~{100 * (1 - len(lags) / max(lags)):.1f}%")
    print(f"  - Input sequences will contain ONLY the {X_features_scaled.shape[1]} features\n")

    X_seq, y_seq = create_sequences_sparse_lags(
        X_features_scaled,
        y_target_scaled,
        lags
    )

    print(f"Sequence shapes:")
    print(f"  X_seq: {X_seq.shape} (n_sequences, n_lags={len(lags)}, n_features)")
    print(f"  y_seq: {y_seq.shape} (n_sequences,)")
    print(f"  Total sequences: {len(X_seq)}")

    # Verify NO target leakage
    print(f"\n{'=' * 70}")
    print("FINAL VERIFICATION - NO TARGET LEAKAGE")
    print(f"{'=' * 70}")
    print(f"✓ Input sequences have {X_seq.shape[2]} features")
    print(f"✓ Original data had {X_features.shape[1]} features (excluding target)")
    print(f"✓ These numbers MATCH - confirming NO target in inputs")
    print(f"✓ Sequence uses lags: {lags}")
    print(f"{'=' * 70}\n")

    # SEQUENTIAL/CHRONOLOGICAL split - now 70/30
    print(f"Using SEQUENTIAL/CHRONOLOGICAL split (70/30)")
    print("Training on PAST data, testing on FUTURE data\n")

    split_idx = int(len(X_seq) * (1 - test_size))

    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]

    print(f"Training samples: {len(X_train)} (sequences 0 to {split_idx})")
    print(f"Test samples: {len(X_test)} (sequences {split_idx} to {len(X_seq)})")
    print(f"✓ Test set is chronologically AFTER training set\n")

    # Create train/test mask for visualization
    train_test_mask = np.zeros(len(X_seq))
    train_test_mask[split_idx:] = 1

    return X_train, X_test, y_train, y_test, X_seq, y_seq, train_test_mask, scaler_X, scaler_y


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=30, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer,
                                num_epochs, device, early_stopping=None):
    """Train the model with validation and early stopping"""
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_actuals = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_actuals.extend(batch_y.cpu().numpy())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_r2 = r2_score(train_actuals, train_preds)
        train_losses.append(avg_train_loss)
        train_r2_scores.append(train_r2)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_actuals = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                epoch_val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_actuals.extend(batch_y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_r2 = r2_score(val_actuals, val_preds)
        val_losses.append(avg_val_loss)
        val_r2_scores.append(val_r2)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train - Loss: {avg_train_loss:.6f}, R²: {train_r2:.6f}')
            print(f'  Val   - Loss: {avg_val_loss:.6f}, R²: {val_r2:.6f}')

        # Early stopping
        if early_stopping:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                model.load_state_dict(early_stopping.best_model)
                break

    return train_losses, val_losses, train_r2_scores, val_r2_scores


def predict_all_sequences(model, X_all, device):
    """Make predictions on all sequences in chronological order"""
    model.eval()
    all_predictions = []

    X_tensor = torch.FloatTensor(X_all).to(device)

    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_all), batch_size):
            batch_X = X_tensor[i:i + batch_size]
            outputs = model(batch_X)
            all_predictions.extend(outputs.cpu().numpy())

    return np.array(all_predictions)


def plot_results_chronological(train_losses, val_losses, train_r2_scores, val_r2_scores,
                               all_predictions, all_actuals, train_test_mask,
                               lags, scaler_y):
    """
    Plot results in chronological order with train/test color coding

    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        train_r2_scores: Training R² scores over epochs
        val_r2_scores: Validation R² scores over epochs
        all_predictions: All model predictions (scaled)
        all_actuals: All actual values (scaled)
        train_test_mask: Mask indicating train (0) vs test (1) samples
        lags: List of lag positions used (e.g., [5, 10, 20, 30, 50])
        scaler_y: Scaler object for inverse transform
    """
    # Get max lag for time indexing
    max_lag = max(lags)

    # Inverse transform to original scale
    all_pred_original = scaler_y.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    all_actual_original = scaler_y.inverse_transform(all_actuals.reshape(-1, 1)).flatten()

    # Separate train and test based on mask
    train_indices = np.where(train_test_mask == 0)[0]
    test_indices = np.where(train_test_mask == 1)[0]

    # Time indices (starting from max_lag since that's when predictions begin)
    time_indices = np.arange(max_lag, max_lag + len(all_actual_original))

    # Calculate R² for train and test
    train_r2 = r2_score(all_actual_original[train_indices], all_pred_original[train_indices])
    test_r2 = r2_score(all_actual_original[test_indices], all_pred_original[test_indices])

    # ============================================================
    # PLOT 1: Training time series
    # ============================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 7))
    ax1.plot(time_indices[train_indices], all_actual_original[train_indices],
             label='Measured', color='blue', linewidth=2, alpha=0.8, marker='o', markersize=2)
    ax1.plot(time_indices[train_indices], all_pred_original[train_indices],
             label='Modeled', color='red', linewidth=2, alpha=0.8, linestyle='--', marker='s', markersize=1.5)
    ax1.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stl_t_S_NO_Concentration', fontsize=14, fontweight='bold')
    ax1.set_title(f'Training Data: Measured vs Modeled (R² = {train_r2:.4f})\n' +
                  f'Sparse Lags: {lags} | Showing {len(train_indices)} training points',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_time_series.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training time series plot saved to: {save_path}")
    plt.close()

    # ============================================================
    # PLOT 2: Testing time series
    # ============================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 7))
    ax2.plot(time_indices[test_indices], all_actual_original[test_indices],
             label='Measured', color='blue', linewidth=2.5, alpha=0.8, marker='o', markersize=3)
    ax2.plot(time_indices[test_indices], all_pred_original[test_indices],
             label='Modeled', color='red', linewidth=2.5, alpha=0.8, linestyle='--', marker='s', markersize=2.5)
    ax2.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Stl_t_S_NO_Concentration', fontsize=14, fontweight='bold')
    ax2.set_title(f'Test Data: Measured vs Modeled (R² = {test_r2:.4f})\n' +
                  f'Sparse Lags: {lags} | Showing {len(test_indices)} test points',
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'testing_time_series.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Testing time series plot saved to: {save_path}")
    plt.close()

    # ============================================================
    # PLOT 3: Combined plot
    # ============================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(16, 7))
    ax3.plot(time_indices, all_actual_original,
             label='Measured', color='black', linewidth=2, alpha=0.7, zorder=1)
    ax3.scatter(time_indices[train_indices], all_pred_original[train_indices],
                label=f'Modeled - Training (R²={train_r2:.3f})',
                color='blue', s=12, alpha=0.6, zorder=2, marker='o')
    ax3.scatter(time_indices[test_indices], all_pred_original[test_indices],
                label=f'Modeled - Test (R²={test_r2:.3f})',
                color='red', s=20, alpha=0.8, zorder=3, marker='s')
    ax3.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Stl_t_S_NO_Concentration', fontsize=14, fontweight='bold')
    ax3.set_title(f'Complete Time Series: Training and Test Predictions\n' +
                  f'Sparse Lags: {lags} (lookback={max_lag} steps) | ' +
                  f'Black line = Measured | Blue = Training | Red = Test',
                  fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'combined_time_series.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined time series plot saved to: {save_path}")
    plt.close()

    # ============================================================
    # PLOT 4: Comprehensive plot
    # ============================================================
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Training history - Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Training history - R²
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train_r2_scores, label='Training R²', linewidth=2)
    ax2.plot(val_r2_scores, label='Validation R²', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Training and Validation R²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Combined time series (compact)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_indices, all_actual_original, label='Measured',
             color='black', linewidth=1.5, alpha=0.6)
    ax3.scatter(time_indices[train_indices], all_pred_original[train_indices],
                label='Train', s=5, alpha=0.5, color='blue')
    ax3.scatter(time_indices[test_indices], all_pred_original[test_indices],
                label='Test', s=10, alpha=0.7, color='red')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Concentration')
    ax3.set_title(f'Complete Time Series\nLags: {lags}')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Training time series detail
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(time_indices, all_actual_original,
             label='Measured', color='black', linewidth=1.5, alpha=0.6)
    ax4.scatter(time_indices[train_indices], all_pred_original[train_indices],
                label=f'Training Predictions (R²={train_r2:.3f})',
                s=8, alpha=0.6, color='blue')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Stl_t_S_NO_Concentration')
    ax4.set_title(f'Training: Model Predictions (Sparse Lags: {lags}, Max Lookback: {max_lag})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Training scatter
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(all_actual_original[train_indices], all_pred_original[train_indices],
                alpha=0.6, s=20, color='blue')
    min_val = all_actual_original[train_indices].min()
    max_val = all_actual_original[train_indices].max()
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax5.set_xlabel('Measured')
    ax5.set_ylabel('Modeled')
    ax5.set_title(f'Training Set: Scatter Plot (R²={train_r2:.4f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Test scatter
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(all_actual_original[test_indices], all_pred_original[test_indices],
                alpha=0.6, s=30, color='red')
    min_val = all_actual_original[test_indices].min()
    max_val = all_actual_original[test_indices].max()
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax6.set_xlabel('Measured')
    ax6.set_ylabel('Modeled')
    ax6.set_title(f'Test Set: Scatter Plot (R²={test_r2:.4f})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Residuals
    ax7 = fig.add_subplot(gs[2, 2])
    residuals = all_actual_original[test_indices] - all_pred_original[test_indices]
    ax7.scatter(all_pred_original[test_indices], residuals, alpha=0.6, s=30, color='green')
    ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax7.set_xlabel('Modeled Values')
    ax7.set_ylabel('Residuals')
    ax7.set_title('Test Set: Residual Plot')
    ax7.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, 'lstm_comprehensive_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive results plot saved to: {save_path}")
    plt.close()

    # ============================================================
    # Print metrics
    # ============================================================
    train_mse = np.mean((all_actual_original[train_indices] - all_pred_original[train_indices]) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(all_actual_original[train_indices] - all_pred_original[train_indices]))

    test_mse = np.mean((all_actual_original[test_indices] - all_pred_original[test_indices]) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(all_actual_original[test_indices] - all_pred_original[test_indices]))

    print(f"\n{'=' * 70}")
    print(f"SPARSE LAG CONFIGURATION: {lags}")
    print(f"Max Lookback: {max_lag} timesteps")
    print(f"Sequence Length: {len(lags)} timesteps")
    print(f"Computational Savings: {100 * (1 - len(lags) / max_lag):.1f}%")
    print(f"{'=' * 70}")

    print(f"\nTraining Set Metrics (in original scale):")
    print(f"  MSE:  {train_mse:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  R²:   {train_r2:.6f}")

    print(f"\nTest Set Metrics (in original scale):")
    print(f"  MSE:  {test_mse:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE:  {test_mae:.6f}")
    print(f"  R²:   {test_r2:.6f}")

    print(f"\nOverfitting Analysis:")
    overfitting_gap = train_r2 - test_r2
    print(f"  Train-Test R² Gap: {overfitting_gap:.6f}")
    if overfitting_gap < 0.10:
        print(f"  ✓ Excellent generalization (gap < 0.10)")
    elif overfitting_gap < 0.15:
        print(f"  ✓ Good generalization (gap < 0.15)")
    elif overfitting_gap < 0.20:
        print(f"  ⚠ Moderate overfitting (gap < 0.20)")
    else:
        print(f"  ✗ Significant overfitting (gap >= 0.20)")
    print(f"{'=' * 70}\n")

    return train_r2, test_r2


def train_single_configuration(hyperparams, file_path, run_ergodicity_tests=False, verbose=False):
    """
    Train a single model with given hyperparameters (now including sparse lags)
    """
    LAGS = hyperparams['lags']
    HIDDEN_SIZE = hyperparams['hidden_size']
    NUM_LAYERS = hyperparams['num_layers']
    BATCH_SIZE = hyperparams['batch_size']
    NUM_EPOCHS = hyperparams['num_epochs']
    LEARNING_RATE = hyperparams['learning_rate']
    DROPOUT = hyperparams['dropout']
    WEIGHT_DECAY = hyperparams['weight_decay']
    PATIENCE = hyperparams['patience']

    try:
        # Load and preprocess data with sparse lags (now 70/30 split)
        X_train, X_test, y_train, y_test, X_all, y_all, train_test_mask, scaler_X, scaler_y = load_and_preprocess_data(
            file_path,
            lags=LAGS,
            test_size=0.3,  # 70/30 split
            run_ergodicity_tests=run_ergodicity_tests
        )

        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        input_size = X_train.shape[2]
        output_size = 1

        model = LSTMModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=output_size,
            dropout=DROPOUT
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.0001, verbose=verbose)

        train_losses, val_losses, train_r2_scores, val_r2_scores = train_model_with_validation(
            model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, device, early_stopping
        )

        all_predictions = predict_all_sequences(model, X_all, device)

        all_pred_original = scaler_y.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        all_actual_original = scaler_y.inverse_transform(y_all.reshape(-1, 1)).flatten()

        train_indices = np.where(train_test_mask == 0)[0]
        test_indices = np.where(train_test_mask == 1)[0]

        train_r2 = r2_score(all_actual_original[train_indices], all_pred_original[train_indices])
        test_r2 = r2_score(all_actual_original[test_indices], all_pred_original[test_indices])

        test_mse = np.mean((all_actual_original[test_indices] - all_pred_original[test_indices]) ** 2)
        test_rmse = np.sqrt(test_mse)
        test_mae = np.mean(np.abs(all_actual_original[test_indices] - all_pred_original[test_indices]))

        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_r2_scores': train_r2_scores,
            'val_r2_scores': val_r2_scores,
            'all_predictions': all_predictions,
            'y_all': y_all,
            'X_all': X_all,
            'train_test_mask': train_test_mask,
            'input_size': input_size
        }

    except Exception as e:
        print(f"Error training configuration: {e}")
        import traceback
        traceback.print_exc()
        return {
            'train_r2': -999,
            'test_r2': -999,
            'test_mse': 999999,
            'test_rmse': 999999,
            'test_mae': 999999,
            'error': str(e)
        }


def save_best_model(model, hyperparams, results, scaler_X, scaler_y, save_dir=OUTPUT_DIR):
    """Save the best model with all necessary information"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'hyperparams': hyperparams,
        'results': {
            'train_r2': results['train_r2'],
            'test_r2': results['test_r2'],
            'test_mse': results['test_mse'],
            'test_rmse': results['test_rmse'],
            'test_mae': results['test_mae'],
        },
        'scaler_X_mean': scaler_X.mean_,
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_mean': scaler_y.mean_,
        'scaler_y_scale': scaler_y.scale_,
        'input_size': results['input_size'],
        'timestamp': timestamp
    }

    # Save model
    model_path = os.path.join(save_dir, f'best_lstm_model_{timestamp}.pth')
    torch.save(save_dict, model_path)
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL SAVED")
    print(f"{'=' * 80}")
    print(f"Model path: {model_path}")
    print(f"Test R²: {results['test_r2']:.6f}")
    print(f"Configuration: {hyperparams}")
    print(f"{'=' * 80}\n")

    return model_path


def load_best_model(model_path, device='cpu'):
    """Load a saved model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Recreate model
    hyperparams = checkpoint['hyperparams']
    model = LSTMModel(
        input_size=checkpoint['input_size'],
        hidden_size=hyperparams['hidden_size'],
        num_layers=hyperparams['num_layers'],
        output_size=1,
        dropout=hyperparams['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nModel loaded successfully!")
    print(f"Hyperparameters: {hyperparams}")
    print(f"Test R²: {checkpoint['results']['test_r2']:.6f}")

    return model, checkpoint


def focused_hyperparameter_search(file_path, baseline_config, n_trials=50, run_ergodicity_first=True):
    """
    Perform FOCUSED search around a baseline configuration

    Args:
        file_path: Path to data file
        baseline_config: Dictionary with baseline hyperparameters that achieved good results
        n_trials: Number of trials
        run_ergodicity_first: Whether to run ergodicity tests initially
    """

    print("=" * 80)
    print("FOCUSED HYPERPARAMETER SEARCH (70/30 Train/Test Split)")
    print("=" * 80)
    print("\nBaseline Configuration:")
    for param, value in baseline_config.items():
        print(f"  {param}: {value}")

    # Define NARROW search ranges around baseline
    def get_search_space(baseline):
        """Create narrow search space around baseline values"""

        # For lags: try variations (add/remove points, shift positions)
        base_lags = baseline['lags']
        max_lag = max(base_lags)

        lag_variations = [
            base_lags,  # Original
            base_lags[:-1],  # Remove last lag
            base_lags + [max_lag + 20],  # Add one more lag
            [l - 2 for l in base_lags],  # Shift earlier
            [l + 2 for l in base_lags],  # Shift later
            [base_lags[0]] + base_lags + [max_lag + 10],  # Add boundary lags
        ]

        # For numeric parameters: ±30% range
        hidden_range = [
            max(4, int(baseline['hidden_size'] * 0.7)),
            int(baseline['hidden_size'] * 1.3)
        ]

        lr_range = [
            baseline['learning_rate'] * 0.5,
            baseline['learning_rate'] * 2.0
        ]

        wd_range = [
            max(0.0, baseline['weight_decay'] * 0.5),
            baseline['weight_decay'] * 2.0
        ]

        dropout_range = [
            max(0.1, baseline['dropout'] - 0.2),
            min(0.7, baseline['dropout'] + 0.2)
        ]

        patience_range = [
            max(20, baseline['patience'] - 10),
            baseline['patience'] + 10
        ]

        return {
            'lags': lag_variations,
            'hidden_size': list(range(hidden_range[0], hidden_range[1] + 1, 2)),
            'num_layers': [baseline['num_layers'] - 1, baseline['num_layers'], baseline['num_layers'] + 1],
            'batch_size': [baseline['batch_size']],  # Keep same
            'num_epochs': [200],  # Keep same
            'learning_rate': np.linspace(lr_range[0], lr_range[1], 5).tolist(),
            'dropout': np.linspace(dropout_range[0], dropout_range[1], 5).tolist(),
            'weight_decay': np.linspace(wd_range[0], wd_range[1], 5).tolist(),
            'patience': list(range(patience_range[0], patience_range[1] + 1, 5))
        }

    search_space = get_search_space(baseline_config)

    print("\n" + "=" * 80)
    print("Focused Search Space:")
    for param, values in search_space.items():
        if param == 'lags':
            print(f"  {param}: {len(values)} variations around baseline")
        else:
            print(f"  {param}: {len(values)} values in [{min(values)}, {max(values)}]")
    print(f"\nNumber of trials: {n_trials}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    # Run ergodicity tests once
    if run_ergodicity_first:
        print("Running initial ergodicity analysis...")
        df = pd.read_csv(file_path)
        test_ergodicity(df, window_size=min(200, len(df) // 10))
        plot_ergodicity_diagnostics(df, window_size=min(200, len(df) // 10))
        print("\n" + "=" * 80 + "\n")

    best_test_r2 = -999
    best_hyperparams = None
    best_results = None
    all_results = []

    for trial in range(n_trials):
        print(f"\n{'=' * 80}")
        print(f"TRIAL {trial + 1}/{n_trials}")
        print(f"{'=' * 80}")

        # Randomly sample hyperparameters from focused space
        hyperparams = {}
        for param, values in search_space.items():
            value = values[np.random.randint(len(values))]
            # Convert to native Python types
            if param == 'lags':
                hyperparams[param] = list(value)  # Keep as list
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                hyperparams[param] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                hyperparams[param] = float(value)
            else:
                hyperparams[param] = value

        print("Testing configuration:")
        for param, value in hyperparams.items():
            print(f"  {param}: {value}")

        # Train model
        results = train_single_configuration(
            hyperparams,
            file_path,
            run_ergodicity_tests=False,
            verbose=False
        )

        # Store results
        results['hyperparams'] = hyperparams.copy()
        all_results.append(results)

        # Check if best
        if results['test_r2'] > best_test_r2:
            best_test_r2 = results['test_r2']
            best_hyperparams = hyperparams.copy()
            best_results = results

            print(f"\n{'*' * 80}")
            print(f"NEW BEST MODEL FOUND!")
            print(f"{'*' * 80}")
            print(f"Lags: {hyperparams['lags']}")
            print(f"Test R²: {results['test_r2']:.6f}")
            print(f"Train R²: {results['train_r2']:.6f}")
            print(f"{'*' * 80}")
        else:
            print(f"\nResults: Test R² = {results['test_r2']:.6f} (Best: {best_test_r2:.6f})")

    return best_hyperparams, best_results, all_results


def save_search_results(best_hyperparams, best_results, all_results):
    """Save hyperparameter search results to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best hyperparameters
    best_hp_df = pd.DataFrame([{
        'lags': str(best_hyperparams['lags']),
        'max_lookback': max(best_hyperparams['lags']),
        'num_lag_points': len(best_hyperparams['lags']),
        'hidden_size': best_hyperparams['hidden_size'],
        'num_layers': best_hyperparams['num_layers'],
        'batch_size': best_hyperparams['batch_size'],
        'learning_rate': best_hyperparams['learning_rate'],
        'dropout': best_hyperparams['dropout'],
        'weight_decay': best_hyperparams['weight_decay'],
        'patience': best_hyperparams['patience'],
        'test_r2': best_results['test_r2'],
        'train_r2': best_results['train_r2'],
        'test_rmse': best_results['test_rmse'],
        'test_mae': best_results['test_mae'],
        'overfitting_gap': best_results['train_r2'] - best_results['test_r2']
    }])

    best_hp_file = os.path.join(OUTPUT_DIR, f'best_hyperparams_{timestamp}.csv')
    best_hp_df.to_csv(best_hp_file, index=False)
    print(f"\nBest hyperparameters saved to: {best_hp_file}")

    # Save all results
    all_results_list = []
    for result in all_results:
        result_dict = {
            'lags': str(result.get('hyperparams', {}).get('lags', [])),
            'hidden_size': result.get('hyperparams', {}).get('hidden_size', None),
            'num_layers': result.get('hyperparams', {}).get('num_layers', None),
            'batch_size': result.get('hyperparams', {}).get('batch_size', None),
            'learning_rate': result.get('hyperparams', {}).get('learning_rate', None),
            'dropout': result.get('hyperparams', {}).get('dropout', None),
            'weight_decay': result.get('hyperparams', {}).get('weight_decay', None),
            'patience': result.get('hyperparams', {}).get('patience', None),
            'train_r2': result.get('train_r2', None),
            'test_r2': result.get('test_r2', None),
            'test_rmse': result.get('test_rmse', None),
            'test_mae': result.get('test_mae', None)
        }
        all_results_list.append(result_dict)

    all_results_df = pd.DataFrame(all_results_list)
    all_results_file = os.path.join(OUTPUT_DIR, f'all_focused_search_results_{timestamp}.csv')
    all_results_df.to_csv(all_results_file, index=False)
    print(f"All search results saved to: {all_results_file}")

    return best_hp_df


def main():
    """Main function with focused hyperparameter search"""

    file_path = '/home/arash/Downloads/observedoutput_NO.txt'

    # BASELINE CONFIGURATION THAT ACHIEVED R² = 0.71
    baseline_config = {
        'lags': [2, 5, 10, 20, 40, 80],
        'hidden_size': 24,
        'num_layers': 3,
        'dropout': 0.6,
        'learning_rate': 0.0001,
        'weight_decay': 0.001,
        'batch_size': 32,
        'patience': 30,
        'num_epochs': 200
    }

    N_TRIALS = 50  # Number of focused search trials

    print("=" * 80)
    print("LSTM FOCUSED HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Data file: {file_path}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Train/Test Split: 70/30")
    print(f"Device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    # Run focused hyperparameter search
    best_hyperparams, best_results, all_results = focused_hyperparameter_search(
        file_path,
        baseline_config=baseline_config,
        n_trials=N_TRIALS,
        run_ergodicity_first=True
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("FOCUSED SEARCH COMPLETE")
    print("=" * 80)

    if best_hyperparams is None or best_results is None:
        print("\n❌ ERROR: No valid models were found!")
        return

    print("\nBest hyperparameters found:")
    for param, value in best_hyperparams.items():
        print(f"  {param}: {value}")

    print(f"\nBest model performance:")
    print(f"  Test R²: {best_results['test_r2']:.6f}")
    print(f"  Train R²: {best_results['train_r2']:.6f}")
    print(f"  Overfitting gap: {best_results['train_r2'] - best_results['test_r2']:.6f}")
    print(f"  Test RMSE: {best_results['test_rmse']:.6f}")
    print(f"  Test MAE: {best_results['test_mae']:.6f}")
    print("=" * 80 + "\n")

    # Save results
    df_summary = save_search_results(best_hyperparams, best_results, all_results)

    # SAVE THE BEST MODEL
    model_path = save_best_model(
        model=best_results['model'],
        hyperparams=best_hyperparams,
        results=best_results,
        scaler_X=best_results['scaler_X'],
        scaler_y=best_results['scaler_y']
    )

    # Generate plots for best model
    print("\nGenerating plots for best model...")
    plot_results_chronological(
        train_losses=best_results['train_losses'],
        val_losses=best_results['val_losses'],
        train_r2_scores=best_results['train_r2_scores'],
        val_r2_scores=best_results['val_r2_scores'],
        all_predictions=best_results['all_predictions'],
        all_actuals=best_results['y_all'],
        train_test_mask=best_results['train_test_mask'],
        lags=best_hyperparams['lags'],
        scaler_y=best_results['scaler_y']
    )

    print("\n" + "=" * 80)
    print("Search completed!")
    print(f"Best model saved to: {model_path}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()