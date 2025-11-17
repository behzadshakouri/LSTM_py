import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""

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

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def create_sequences_sparse_lags(X_features, y_target, lags):
    """Create sequences using sparse lags"""
    X_seq, y_seq = [], []
    max_lag = max(lags)

    for i in range(max_lag, len(X_features)):
        sequence = np.stack([X_features[i - lag, :] for lag in lags])
        X_seq.append(sequence)
        y_seq.append(y_target[i])

    return np.array(X_seq), np.array(y_seq)


def load_model_and_predict(model_path, data_path, output_dir='/home/arash/Downloads'):
    """
    Load saved model and generate predictions for train and test sets

    Args:
        model_path: Path to saved model (.pth file)
        data_path: Path to original data file
        output_dir: Directory to save CSV files
    """

    print("=" * 80)
    print("LOADING MODEL AND GENERATING PREDICTIONS")
    print("=" * 80)

    # ============================================================
    # 1. LOAD THE SAVED MODEL
    # ============================================================
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract hyperparameters and results
    hyperparams = checkpoint['hyperparams']
    results = checkpoint['results']

    print("\nModel Configuration:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    print(f"\nModel Performance:")
    print(f"  Test R²: {results['test_r2']:.6f}")
    print(f"  Train R²: {results['train_r2']:.6f}")

    # Recreate the model
    model = LSTMModel(
        input_size=checkpoint['input_size'],
        hidden_size=hyperparams['hidden_size'],
        num_layers=hyperparams['num_layers'],
        output_size=1,
        dropout=hyperparams['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("\n✓ Model loaded successfully!")

    # ============================================================
    # 2. LOAD AND PREPROCESS DATA
    # ============================================================
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")

    # Get the lags from hyperparams
    lags = hyperparams['lags']
    print(f"Using lags: {lags}")

    # Separate features and target
    data = df.values
    X_features = data[:, :-1]
    y_target = data[:, -1]

    print(f"\nFeature shape: {X_features.shape}")
    print(f"Target shape: {y_target.shape}")

    # Recreate scalers from saved parameters
    scaler_X = StandardScaler()
    scaler_X.mean_ = checkpoint['scaler_X_mean']
    scaler_X.scale_ = checkpoint['scaler_X_scale']
    scaler_X.var_ = checkpoint['scaler_X_scale'] ** 2
    scaler_X.n_features_in_ = len(checkpoint['scaler_X_mean'])

    scaler_y = StandardScaler()
    scaler_y.mean_ = checkpoint['scaler_y_mean']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    scaler_y.var_ = checkpoint['scaler_y_scale'] ** 2
    scaler_y.n_features_in_ = 1

    # Scale the data
    X_features_scaled = scaler_X.transform(X_features)
    y_target_scaled = scaler_y.transform(y_target.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq, y_seq = create_sequences_sparse_lags(
        X_features_scaled,
        y_target_scaled,
        lags
    )

    print(f"\nSequence shapes:")
    print(f"  X_seq: {X_seq.shape}")
    print(f"  y_seq: {y_seq.shape}")

    # Split into train/test (same 70/30 split as training)
    split_idx = int(len(X_seq) * 0.7)

    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # ============================================================
    # 3. MAKE PREDICTIONS
    # ============================================================
    print("\nGenerating predictions...")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        # Predict in batches
        batch_size = 64

        # Training predictions
        train_predictions = []
        for i in range(0, len(X_train_tensor), batch_size):
            batch = X_train_tensor[i:i + batch_size]
            pred = model(batch)
            train_predictions.extend(pred.cpu().numpy())
        train_predictions = np.array(train_predictions).flatten()

        # Test predictions
        test_predictions = []
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i + batch_size]
            pred = model(batch)
            test_predictions.extend(pred.cpu().numpy())
        test_predictions = np.array(test_predictions).flatten()

    # Inverse transform to original scale
    train_pred_original = scaler_y.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
    train_actual_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()

    test_pred_original = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    test_actual_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    print("✓ Predictions generated successfully!")

    # ============================================================
    # 4. CREATE CSV FILES
    # ============================================================
    print("\nCreating CSV files...")

    # Get max lag for time indexing
    max_lag = max(lags)

    # Training CSV
    train_df = pd.DataFrame({
        'Time_Step': np.arange(max_lag, max_lag + len(train_actual_original)),
        'Actual_Concentration': train_actual_original,
        'Predicted_Concentration': train_pred_original,
        'Residual': train_actual_original - train_pred_original,
        'Absolute_Error': np.abs(train_actual_original - train_pred_original),
        'Squared_Error': (train_actual_original - train_pred_original) ** 2,
        'Percentage_Error': 100 * np.abs(train_actual_original - train_pred_original) / np.abs(train_actual_original)
    })

    train_csv_path = f"{output_dir}/train_predictions.csv"
    train_df.to_csv(train_csv_path, index=False)
    print(f"✓ Training predictions saved to: {train_csv_path}")

    # Test CSV
    test_df = pd.DataFrame({
        'Time_Step': np.arange(max_lag + len(train_actual_original),
                               max_lag + len(train_actual_original) + len(test_actual_original)),
        'Actual_Concentration': test_actual_original,
        'Predicted_Concentration': test_pred_original,
        'Residual': test_actual_original - test_pred_original,
        'Absolute_Error': np.abs(test_actual_original - test_pred_original),
        'Squared_Error': (test_actual_original - test_pred_original) ** 2,
        'Percentage_Error': 100 * np.abs(test_actual_original - test_pred_original) / np.abs(test_actual_original)
    })

    test_csv_path = f"{output_dir}/test_predictions.csv"
    test_df.to_csv(test_csv_path, index=False)
    print(f"✓ Test predictions saved to: {test_csv_path}")

    # Combined CSV
    combined_df = pd.concat([
        train_df.assign(Dataset='Train'),
        test_df.assign(Dataset='Test')
    ], ignore_index=True)

    combined_csv_path = f"{output_dir}/combined_predictions.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Combined predictions saved to: {combined_csv_path}")

    # ============================================================
    # 5. PRINT SUMMARY STATISTICS
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # Training metrics
    train_r2 = r2_score(train_actual_original, train_pred_original)
    train_rmse = np.sqrt(mean_squared_error(train_actual_original, train_pred_original))
    train_mae = mean_absolute_error(train_actual_original, train_pred_original)

    print("\nTRAINING SET:")
    print(f"  R² Score: {train_r2:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    print(f"  Mean Actual: {np.mean(train_actual_original):.6f}")
    print(f"  Mean Predicted: {np.mean(train_pred_original):.6f}")
    print(f"  Std Actual: {np.std(train_actual_original):.6f}")
    print(f"  Std Predicted: {np.std(train_pred_original):.6f}")

    # Test metrics
    test_r2 = r2_score(test_actual_original, test_pred_original)
    test_rmse = np.sqrt(mean_squared_error(test_actual_original, test_pred_original))
    test_mae = mean_absolute_error(test_actual_original, test_pred_original)

    print("\nTEST SET:")
    print(f"  R² Score: {test_r2:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  Mean Actual: {np.mean(test_actual_original):.6f}")
    print(f"  Mean Predicted: {np.mean(test_pred_original):.6f}")
    print(f"  Std Actual: {np.std(test_actual_original):.6f}")
    print(f"  Std Predicted: {np.std(test_pred_original):.6f}")

    print("\n" + "=" * 80)
    print("CSV FILES CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n📁 Files saved to: {output_dir}")
    print(f"  1. {train_csv_path}")
    print(f"  2. {test_csv_path}")
    print(f"  3. {combined_csv_path}")
    print("\n" + "=" * 80)

    return train_df, test_df, combined_df


def main():
    """Main function"""

    # File paths
    MODEL_PATH = '/home/arash/Downloads/NO_outputs/best_lstm_model_20251113_140159.pth'
    DATA_PATH = '/home/arash/Downloads/observedoutput_NO.txt'
    OUTPUT_DIR = '/home/arash/Downloads/NO_outputs'

    # Load model and generate predictions
    train_df, test_df, combined_df = load_model_and_predict(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )

    # Display first few rows
    print("\nSample of Training Predictions (first 5 rows):")
    print(train_df.head().to_string())

    print("\nSample of Test Predictions (first 5 rows):")
    print(test_df.head().to_string())


if __name__ == "__main__":
    main()