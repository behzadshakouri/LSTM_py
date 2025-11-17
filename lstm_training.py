import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# LSTM Model Definition
# ============================================================================
class SparseLagLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(SparseLagLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ============================================================================
# Dataset Class
# ============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# Sparse Lag Sequence Creation
# ============================================================================
def create_sparse_lag_sequences(X_features, y_target, lags):
    """
    Create sequences using sparse lags
    """
    max_lag = max(lags)
    n_samples = len(X_features)
    n_features = X_features.shape[1]
    n_lags = len(lags)

    # Pre-allocate arrays
    X_sequences = np.zeros((n_samples - max_lag, n_lags, n_features))
    y_sequences = np.zeros(n_samples - max_lag)

    # Create sequences
    for i in range(max_lag, n_samples):
        for j, lag in enumerate(lags):
            X_sequences[i - max_lag, j, :] = X_features[i - lag]
        y_sequences[i - max_lag] = y_target[i]

    return X_sequences, y_sequences


# ============================================================================
# Training Function
# ============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, patience, device):
    """Train the model with early stopping"""

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '/home/arash/Downloads/best_model_final.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                break

    # Load best model
    model.load_state_dict(torch.load('/home/arash/Downloads/best_model_final.pth'))

    return model, train_losses, val_losses


# ============================================================================
# Prediction Function
# ============================================================================
def make_predictions(model, data_loader, device):
    """Make predictions on a dataset"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())

    return np.array(predictions), np.array(actuals)


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    print("=" * 80)
    print("TRAINING BEST LSTM CONFIGURATION")
    print("=" * 80)

    # Best hyperparameters from search
    BEST_CONFIG = {
        'lags': [2, 5, 10, 20, 40, 80],
        'hidden_size': 24,
        'num_layers': 3,
        'batch_size': 32,
        'num_epochs': 200,
        'learning_rate': 0.0001,
        'dropout': 0.6,
        'weight_decay': 0.001,
        'patience': 30
    }

    print("\nConfiguration:")
    for key, value in BEST_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data_file = '/home/arash/Downloads/observedoutput_NO.txt'
    df = pd.read_csv(data_file)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Separate features and target
    X_features = df.iloc[:, :-1].values  # All columns except last
    y_target = df.iloc[:, -1].values  # Last column (NO concentration)
    time_column = df.iloc[:, 0].values  # First column (time)

    # Normalize features and target separately
    print("\nNormalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_features)
    y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).flatten()

    # Create sparse lag sequences
    print(f"\nCreating sparse lag sequences with lags: {BEST_CONFIG['lags']}")
    X_seq, y_seq = create_sparse_lag_sequences(X_scaled, y_scaled, BEST_CONFIG['lags'])

    print(f"Sequence shapes:")
    print(f"  X_seq: {X_seq.shape}")
    print(f"  y_seq: {y_seq.shape}")

    # Adjust time column to match sequences
    max_lag = max(BEST_CONFIG['lags'])
    time_seq = time_column[max_lag:]

    # Split data (80/20 chronological split)
    split_idx = int(0.8 * len(X_seq))

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    time_train, time_test = time_seq[:split_idx], time_seq[split_idx:]

    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BEST_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BEST_CONFIG['batch_size'], shuffle=False)

    # Initialize model
    print("\nInitializing model...")
    n_features = X_seq.shape[2]
    model = SparseLagLSTM(
        input_size=n_features,
        hidden_size=BEST_CONFIG['hidden_size'],
        num_layers=BEST_CONFIG['num_layers'],
        dropout=BEST_CONFIG['dropout']
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=BEST_CONFIG['learning_rate'],
        weight_decay=BEST_CONFIG['weight_decay']
    )

    # Train model
    print("\nTraining model...")
    print("=" * 80)
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        BEST_CONFIG['num_epochs'], BEST_CONFIG['patience'], device
    )

    print("=" * 80)
    print("Training complete!")

    # Make predictions
    print("\nMaking predictions...")
    train_pred, train_actual = make_predictions(model, train_loader, device)
    test_pred, test_actual = make_predictions(model, test_loader, device)

    # Inverse transform predictions
    train_pred_original = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
    train_actual_original = scaler_y.inverse_transform(train_actual.reshape(-1, 1)).flatten()
    test_pred_original = scaler_y.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    test_actual_original = scaler_y.inverse_transform(test_actual.reshape(-1, 1)).flatten()

    # Calculate metrics
    train_r2 = 1 - np.sum((train_actual_original - train_pred_original) ** 2) / \
               np.sum((train_actual_original - train_actual_original.mean()) ** 2)
    test_r2 = 1 - np.sum((test_actual_original - test_pred_original) ** 2) / \
              np.sum((test_actual_original - test_actual_original.mean()) ** 2)
    test_rmse = np.sqrt(np.mean((test_actual_original - test_pred_original) ** 2))
    test_mae = np.mean(np.abs(test_actual_original - test_pred_original))

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Training R²:   {train_r2:.6f}")
    print(f"Test R²:       {test_r2:.6f}")
    print(f"Test RMSE:     {test_rmse:.6f}")
    print(f"Test MAE:      {test_mae:.6f}")
    print(f"Overfitting:   {train_r2 - test_r2:.6f}")
    print("=" * 80)

    # Save results to CSV
    print("\nSaving results to CSV...")

    # Combine train and test results
    all_time = np.concatenate([time_train, time_test])
    all_actual = np.concatenate([train_actual_original, test_actual_original])
    all_predicted = np.concatenate([train_pred_original, test_pred_original])
    all_dataset = np.concatenate([np.full(len(time_train), 'train'),
                                  np.full(len(time_test), 'test')])

    # Create DataFrame
    results_df = pd.DataFrame({
        'Time': all_time,
        'Actual_NO': all_actual,
        'Predicted_NO': all_predicted,
        'Residual': all_actual - all_predicted,
        'Dataset': all_dataset
    })

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'/home/arash/Downloads/NO_predictions_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Full time series
    ax1 = axes[0, 0]
    ax1.plot(time_train, train_actual_original, 'b-', label='Actual (Train)', alpha=0.7, linewidth=1)
    ax1.plot(time_train, train_pred_original, 'r-', label='Predicted (Train)', alpha=0.7, linewidth=1)
    ax1.plot(time_test, test_actual_original, 'b-', label='Actual (Test)', alpha=0.7, linewidth=1)
    ax1.plot(time_test, test_pred_original, 'g-', label='Predicted (Test)', alpha=0.7, linewidth=1)
    ax1.axvline(x=time_test[0], color='k', linestyle='--', linewidth=1, label='Train/Test Split')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('NO Concentration')
    ax1.set_title('NO Concentration: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test set detail
    ax2 = axes[0, 1]
    ax2.plot(time_test, test_actual_original, 'b-', label='Actual', linewidth=2)
    ax2.plot(time_test, test_pred_original, 'r--', label='Predicted', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('NO Concentration')
    ax2.set_title(f'Test Set Detail (R² = {test_r2:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter plot
    ax3 = axes[1, 0]
    ax3.scatter(test_actual_original, test_pred_original, alpha=0.5, s=20)
    min_val = min(test_actual_original.min(), test_pred_original.min())
    max_val = max(test_actual_original.max(), test_pred_original.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual NO Concentration')
    ax3.set_ylabel('Predicted NO Concentration')
    ax3.set_title(f'Test Set: Actual vs Predicted (R² = {test_r2:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals
    ax4 = axes[1, 1]
    residuals = test_actual_original - test_pred_original
    ax4.scatter(time_test, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Residuals')
    ax4.set_title(f'Prediction Residuals (MAE = {test_mae:.4f})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = f'/home/arash/Downloads/NO_predictions_plot_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()

    # Save summary statistics
    print("\nSaving summary statistics...")
    summary_df = pd.DataFrame({
        'Metric': ['Train R²', 'Test R²', 'Test RMSE', 'Test MAE', 'Overfitting Gap'],
        'Value': [train_r2, test_r2, test_rmse, test_mae, train_r2 - test_r2]
    })

    summary_file = f'/home/arash/Downloads/NO_model_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()