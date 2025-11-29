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


# 1. Predict next day (Model outputs Log-space prediction)
def predict_next_day(model, history_window, device):
    """
    Predict the next day's sales using a history window.
    Output is in Log space if the model was trained on Log data.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(history_window, dtype=torch.float32).unsqueeze(0).to(device)
        y_hat = model(x)
        return y_hat.detach().cpu().numpy().item()


# 2. Recursive 28-day forecast
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

    # 1) True values (Log space because load_ca1_features applies log)
    true_28_log = df_item["sales"].values[-H:]

    # 2) Initial history window
    history = data[N - H - seq_len : N - H].copy()

    preds_log = []

    # Identify sales column index
    try:
        sales_idx = feature_cols.index(data_config.target_col)
    except ValueError:
        sales_idx = feature_cols.index("sales")

    # 3) Roll forward
    for step in range(H):
        target_idx = N - H + step

        # Predict (Log space)
        pred = predict_next_day(model, history, device)
        preds_log.append(pred)

        # Build next row
        next_row = data[target_idx].copy()
        
        # Autoregressive update: feed prediction back into history
        next_row[sales_idx] = pred

        # Update history
        history = np.vstack([history[1:], next_row])

    return preds_log, true_28_log


# 3. Main evaluation
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on last 28 days")
    parser.add_argument("--model", type=str, default="lstm", help="lstm, transformer, transformer_d")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--run-id", type=str, default=None, help="MLflow run ID")
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
    # 1. Load data (Now this returns LOG-transformed sales)
    df = load_ca1_features()

    # 2. Select features (Convert tuple to LIST to avoid KeyError)
    feature_cols = list(data_config.all_features)
    print(f"[Info] Using feature columns: {feature_cols}")
    
    input_dim = len(feature_cols)

    # Load Model
    model = get_model(args.model, input_dim=input_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    print(f"[Info] Model '{args.model}' loaded from {args.checkpoint}")
        
    mlflow.log_param("checkpoint_path", args.checkpoint)
    mlflow.log_param("model", args.model)

    # 3. Test set
    test_df = df[df["date"] > data_config.val_end]
    unique_items = test_df["item_id"].unique()
    
    print(f"[Info] Forecasting for {len(unique_items)} items...")

    # 4. Forecast Loop
    results = []
    pbar = tqdm(unique_items, desc="Forecasting")
    
    for item in pbar:
        df_item = df[df["item_id"] == item].sort_values("date")

        # Get Log-space predictions and truth
        pred_log, true_log = forecast_28_days(
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
                "true_log": float(true_log[i]),
                "pred_log": float(pred_log[i]),
            })
        
        if len(results) % (28 * 10) == 0:
            pbar.set_postfix({"count": len(results)//28})

    df_result = pd.DataFrame(results)

    # Inverse Transform (Log -> Real)
    print("[Eval] Converting Log-space predictions back to Real-space...")
    
    log_tru = df_result["true_log"].values
    log_pred = df_result["pred_log"].values

    # Inverse Log1p is Expm1 (exp(x) - 1)
    tru = np.expm1(log_tru)
    pred = np.expm1(log_pred)
    
    # Clip negative predictions to 0 (sales cannot be negative)
    pred = np.maximum(pred, 0)
    
    # Save Real values back to dataframe for CSV
    df_result["true_real"] = tru
    df_result["pred_real"] = pred

    # Save to CSV
    output_path = args.output or f"{args.model}_28day_forecast.csv"
    df_result.to_csv(output_path, index=False)
    print(f"[Info] Results saved to {output_path}")

    # 5. Calculate Metrics on REAL values
    mae = np.mean(np.abs(pred - tru))
    mse = np.mean((pred - tru) ** 2)
    rmse = np.sqrt(mse)
    
    # Pearson
    if np.std(pred) < 1e-9 or np.std(tru) < 1e-9:
        pearson_r = 0.0
    else:
        pearson_r, _ = pearsonr(tru, pred)

    print(f"\n====== FINAL EVALUATION ({args.model.upper()}) ======")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")
    print("=====================================\n")
    
    # Log metrics
    mlflow.log_metrics({
        "eval_mae": mae,
        "eval_mse": mse,
        "eval_rmse": rmse,
        "eval_pearson_r": pearson_r,
    })
    mlflow.log_artifact(output_path)
    print(f"[Info] Metrics logged. Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()