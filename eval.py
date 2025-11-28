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
    Perform multi-step (28 days) forecast for a single item, aligned to the last 28 days.

    Logic:
    - Use the seq_len days *before* the last 28 days as the initial history.
    - Then roll forward one day at a time, predicting the last 28 days.
    
    Args:
        model: Trained model
        df_item: DataFrame for a single item (full history)
        feature_cols: List of feature column names
        seq_len: Sequence length
        device: torch device
    
    Returns:
        preds_28: list of 28 predicted values
        true_28:  numpy array of 28 true sales values (aligned in time)
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
    #    history covers [N-H-seq_len, ..., N-H-1]
    history = data[N - H - seq_len : N - H].copy()

    preds = []

    # 3) Roll forward from day index N-H to N-1 (these are the last 28 days)
    for step in range(H):
        target_idx = N - H + step  # index of the target day in data

        # Use current history to predict this day's sales
        pred = predict_next_day(model, history, device)
        preds.append(pred)

        # Build next_row features:
        # - Copy real calendar features from that day (date/wday/month/year/...)
        # - Replace sales with our prediction
        next_row = data[target_idx].copy()
        sales_idx = feature_cols.index("sales")
        next_row[sales_idx] = pred

        # Slide window: drop the earliest row, append next_row
        history = np.vstack([history[1:], next_row])

    return preds, true_28


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
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID to log metrics to (if None, creates a new run)",
    )
    args = parser.parse_args()

    device = torch.device(train_config.device)
    print(f"[Info] Using device: {device}")

    # Initialize MLflow
    mlflow.set_experiment("M5_Demand_Forecasting")
    
    # Start MLflow run (either use existing run_id or create new one)
    if args.run_id:
        # Log to existing run
        with mlflow.start_run(run_id=args.run_id):
            _evaluate_model(args, device)
    else:
        # Create new run for evaluation
        with mlflow.start_run(run_name=f"{args.model}_evaluation"):
            _evaluate_model(args, device)


def _evaluate_model(args, device):
    """Internal evaluation function that runs within MLflow context."""
    # 1. Load data (complete CA1 dataset)
    df = load_ca1_features()

    # 2. Load trained model
    feature_cols = list(data_config.simple_features)
    input_dim = len(feature_cols)

    model = get_model(args.model, input_dim=input_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    print(f"[Info] Model '{args.model}' loaded successfully from {args.checkpoint}")
    
    # Log checkpoint path to MLflow
    mlflow.log_param("checkpoint_path", args.checkpoint)
    mlflow.log_param("model", args.model)

    # 3. Extract test set
    test_df = df[df["date"] > data_config.val_end]

    # 4. Forecast 28 days for each item
    results = []

    unique_items = test_df["item_id"].unique()
    print(f"[Info] Forecasting for {len(unique_items)} items...")

    pbar = tqdm(unique_items, desc="Forecasting")
    for item in pbar:
        df_item = df[df["item_id"] == item].sort_values("date")

        # Model predictions and true values for the last 28 days
        pred_28, true_28 = forecast_28_days(
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
                "true": float(true_28[i]),
                "pred": float(pred_28[i]),
            })
        
        pbar.set_postfix({"items": f"{len(results)//28}/{len(unique_items)}"})

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
    
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    # sMAPE = (100/n) * Σ(2 * |actual - forecast| / (|actual| + |forecast|))
    # Avoid division by zero
    smape = np.mean(2 * np.abs(tru - pred) / (np.abs(tru) + np.abs(pred) + 1e-8)) * 100
    
    # Pearson correlation coefficient
    pearson_r, p_value = pearsonr(tru, pred)

    print(f"\n====== FINAL EVALUATION ({args.model.upper()}) ======")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"sMAPE: {smape:.4f}%")
    print(f"Pearson correlation r: {pearson_r:.4f}")
    print("=====================================\n")
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        "eval_mae": mae,
        "eval_rmse": rmse,
        "eval_smape": smape,
        "eval_pearson_r": pearson_r,
    })
    mlflow.log_artifact(output_path)
    print(f"[Info] Evaluation metrics logged to MLflow")
    print(f"[Info] MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()