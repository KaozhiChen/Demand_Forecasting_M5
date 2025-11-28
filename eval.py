# eval.py
import argparse
import torch
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch

from config import data_config, train_config
from data import load_ca1_features
from train import get_model


# 1. Predict next day using history window
def predict_next_day(model, history_window, device):
    """
    Predict the next day's sales using a history window.
    
    Args:
        model: Trained model
        history_window: numpy array, shape (seq_len, num_features)
        device: torch device
    
    Returns:
        Next day's predicted value (scalar)
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(history_window, dtype=torch.float32).unsqueeze(0)
        x = x.to(device)
        y_hat = model(x)
        return y_hat.cpu().numpy().item()


# 2. Recursive 28-day forecast for each item
def forecast_28_days(model, df_item, feature_cols, seq_len, device):
    """
    Perform multi-step (28 days) forecast for a single item.
    
    Args:
        model: Trained model
        df_item: DataFrame for a single item
        feature_cols: List of feature column names
        seq_len: Sequence length
        device: torch device
    
    Returns:
        List of 28 predicted values
    """
    df_item = df_item.sort_values("date")
    data = df_item[feature_cols].values.astype(np.float32)

    predictions = []

    # Use the last seq_len days as input for the first prediction step
    history = data[-seq_len:].copy()

    for _ in range(28):
        # Predict next day
        pred = predict_next_day(model, history, device)
        predictions.append(pred)

        # Update history: remove first day + add new prediction (only update sales column)
        next_row = history[-1].copy()
        next_row[feature_cols.index("sales")] = pred

        history = np.vstack([history[1:], next_row])

    return predictions



# 3. Main evaluation function
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on last 28 days")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        help="Model name: lstm, transformer, etc.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: {model}_28day_forecast.csv)",
    )
    args = parser.parse_args()

    device = torch.device(train_config.device)
    print(f"[Info] Using device: {device}")

    # 1. Load data (complete CA1 dataset)
    df = load_ca1_features()

    # 2. Load trained model
    feature_cols = list(data_config.simple_features)
    input_dim = len(feature_cols)

    model = get_model(args.model, input_dim=input_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    print(f"[Info] Model '{args.model}' loaded successfully from {args.checkpoint}")

    # 3. Extract test set
    test_df = df[df["date"] > data_config.val_end]

    # 4. Forecast 28 days for each item
    results = []

    unique_items = test_df["item_id"].unique()
    print(f"[Info] Forecasting for {len(unique_items)} items...")

    for item in unique_items:
        df_item = df[df["item_id"] == item]

        # True values for the last 28 days
        true_28 = df_item.sort_values("date")["sales"].values[-28:]

        # Model predictions
        pred_28 = forecast_28_days(
            model,
            df_item,
            feature_cols=feature_cols,
            seq_len=data_config.seq_len,
            device=device,
        )

        # Record results
        for i in range(28):
            results.append({
                "item_id": item,
                "day_offset": i + 1,
                "true": true_28[i],
                "pred": pred_28[i],
            })

    df_result = pd.DataFrame(results)

    # Save results
    output_path = args.output or f"{args.model}_28day_forecast.csv"
    df_result.to_csv(output_path, index=False)
    print(f"[Info] Results saved to {output_path}")

    # 5. Calculate error metrics
    tru = df_result["true"].values
    pred = df_result["pred"].values

    mae = np.mean(np.abs(pred - tru))
    rmse = np.sqrt(np.mean((pred - tru) ** 2))
    mape = np.mean(np.abs((pred - tru) / (tru + 1e-6))) * 100

    print(f"\n====== FINAL EVALUATION ({args.model.upper()}) ======")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("=====================================\n")
    
    # Log metrics to MLflow (if running in MLflow context)
    try:
        if mlflow.active_run() is not None:
            mlflow.log_metrics({
                "eval_mae": mae,
                "eval_rmse": rmse,
                "eval_mape": mape,
            })
            mlflow.log_artifact(output_path)
            print(f"[Info] Evaluation metrics logged to MLflow")
    except Exception as e:
        # If not in MLflow context, just continue
        pass


if __name__ == "__main__":
    main()