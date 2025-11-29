# eval.py
import argparse
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from config import data_config, train_config
from data import load_ca1_features
from train import get_model


# 1. Predict next day using history window (no normalization)
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
        x = torch.tensor(history_window, dtype=torch.float32).unsqueeze(0).to(device)
        y_hat = model(x)
        return y_hat.detach().cpu().numpy().item()


# 2. Recursive 28-day forecast for each item
def forecast_28_days(model, df_item, feature_cols, seq_len, device):
    """
    Perform multi-step (28 days) forecast for a single item.
    """
    df_item = df_item.sort_values("date")
    data = df_item[feature_cols].values.astype(np.float32)

    H = 28
    N = len(df_item)

    if N < H + seq_len:
        raise ValueError(
            f"Item {df_item['item_id'].iloc[0]} has too few days ({N}) "
            f"for seq_len={seq_len} and horizon={H}"
        )

    # 1) True values for the last 28 days (used for evaluation)
    true_28 = df_item["sales"].values[-H:]

    # 2) Initial history window: the seq_len days BEFORE the last 28 days
    history = data[N - H - seq_len : N - H].copy()

    preds = []

    # Find the index of the 'sales' column dynamically
    # Use target_col from config to be safe
    try:
        sales_idx = feature_cols.index(data_config.target_col)
    except ValueError:
        # Fallback if target_col is not in features (unlikely)
        sales_idx = feature_cols.index("sales")

    # 3) Roll forward
    for step in range(H):
        target_idx = N - H + step

        # Predict
        pred = predict_next_day(model, history, device)
        preds.append(pred)

        # Build next_row
        next_row = data[target_idx].copy()
        
        # Replace the true sales with our prediction (Autoregressive)
        next_row[sales_idx] = pred

        # Slide window
        history = np.vstack([history[1:], next_row])

    return preds, true_28


# 3. Main evaluation function
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on last 28 days")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        help="Model name: lstm, transformer, transformer_d",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (e.g., lstm_best.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to log metrics to",
    )
    args = parser.parse_args()

    device = torch.device(train_config.device)
    print(f"[Info] Using device: {device}")

    mlflow.set_experiment("M5_Demand_Forecasting")
    
    if args.run_id:
        with mlflow.start_run(run_id=args.run_id):
            _evaluate_model(args, device)
    else:
        with mlflow.start_run(run_name=f"{args.model}_evaluation"):
            _evaluate_model(args, device)


def _evaluate_model(args, device):
    """Internal evaluation function."""
    # 1. Load data
    df = load_ca1_features()

    # 2. Select features (Unified Logic now)
    feature_cols = list(data_config.all_features)
    print(f"[Info] Using feature columns: {feature_cols}")
    
    input_dim = len(feature_cols)

    # Load Model structure
    model = get_model(args.model, input_dim=input_dim)
    
    # Load Weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    print(f"[Info] Model '{args.model}' loaded from {args.checkpoint}")
        
    mlflow.log_param("checkpoint_path", args.checkpoint)
    mlflow.log_param("model", args.model)

    # 3. Extract test set
    test_df = df[df["date"] > data_config.val_end]
    unique_items = test_df["item_id"].unique()
    
    print(f"[Info] Forecasting for {len(unique_items)} items over 28 days...")

    # 4. Forecast
    results = []
    pbar = tqdm(unique_items, desc="Forecasting")
    for item in pbar:
        df_item = df[df["item_id"] == item].sort_values("date")

        pred_28, true_28 = forecast_28_days(
            model,
            df_item,
            feature_cols=feature_cols,
            seq_len=data_config.seq_len,
            device=device,
        )

        for i in range(28):
            results.append({
                "item_id": item,
                "day_offset": i + 1,
                "true": float(true_28[i]),
                "pred": float(pred_28[i]),
            })
        
        # Simple progress update
        if len(results) % (28 * 10) == 0:
            pbar.set_postfix({"count": len(results)//28})

    df_result = pd.DataFrame(results)

    # Save CSV
    output_path = args.output or f"{args.model}_28day_forecast.csv"
    df_result.to_csv(output_path, index=False)
    print(f"[Info] Results saved to {output_path}")

    # 5. Calculate Metrics (Modified: Added MSE, Removed sMAPE)
    tru = df_result["true"].values
    pred = df_result["pred"].values

    # MAE
    mae = np.mean(np.abs(pred - tru))
    
    # MSE & RMSE
    mse = np.mean((pred - tru) ** 2)
    rmse = np.sqrt(mse)
    
    # Pearson
    # Handle case where variation is 0 (constant prediction) to avoid warnings
    if np.std(pred) < 1e-9 or np.std(tru) < 1e-9:
        pearson_r = 0.0
    else:
        pearson_r, _ = pearsonr(tru, pred)

    print(f"\n====== FINAL EVALUATION ({args.model.upper()}) ======")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")   # Added
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")
    print("=====================================\n")
    
    # Log to MLflow
    mlflow.log_metrics({
        "eval_mae": mae,
        "eval_mse": mse,       # Added
        "eval_rmse": rmse,
        "eval_pearson_r": pearson_r,
    })
    mlflow.log_artifact(output_path)
    print(f"[Info] Metrics logged. Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()