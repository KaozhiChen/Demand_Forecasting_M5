# train.py
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np
import random
from config import data_config, train_config
from data import load_ca1_features, create_dataloaders
from models.lstm import LSTM
from models.transformer import Transformer
from models.transformer_d import TransformerD  


# Model selection
def get_model(model_name: str, input_dim: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "lstm":
        model = LSTM(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )
    elif model_name == "transformer":
        # Standard Transformer
        model = Transformer(
            input_dim=input_dim,
            d_model=train_config.d_model,
            nhead=train_config.nhead,
            num_layers=train_config.num_layers,
            dim_feedforward=train_config.dim_feedforward,
            dropout=train_config.dropout,
            max_seq_len=data_config.seq_len + 50,
        )
    elif model_name == "transformer_d":
        raise NotImplementedError("TransformerD code is coming next!")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

# Training Step
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for x, y in pbar:
        x = x.to(device)  # (B, L, F)
        y = y.to(device)  # (B, 1)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader.dataset)

# Validation Step
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="M5 Forecasting Training Script")
    parser.add_argument("--model", type=str, default="lstm", help="lstm, transformer, transformer_d")
    parser.add_argument("--epochs", type=int, default=train_config.epochs)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(train_config.device)
    print(f"[Info] Using device: {device}")

    # 1. Load Data
    print("[Info] Loading data...")
    df_all = load_ca1_features()

    feature_cols = data_config.all_features
    print(f"[Info] Loading features: {feature_cols}")
    print(f"[Info] Total feature count: {len(feature_cols)}")
    
    input_dim = len(feature_cols)

    train_loader, val_loader, test_loader = create_dataloaders(
        df_all,
        feature_cols=feature_cols,
        batch_size=train_config.batch_size,
        seq_len=data_config.seq_len,
        stride=data_config.stride,
    )

    # 2. Model Setup
    print(f"[Info] Building model: {args.model}")
    model = get_model(args.model, input_dim=input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=train_config.lr)
    
    # Early Stopping params
    patience = train_config.early_stop_patience
    min_delta = train_config.early_stop_min_delta
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print("[Info] Starting training...")
    mlflow.set_experiment("M5_Demand_Forecasting")
    
    with mlflow.start_run(run_name=f"{args.model}_{args.epochs}ep"):
        # Log Params
        mlflow.log_params({
            "model": args.model,
            "input_dim": input_dim,
            "feature_list": str(feature_cols), 
            "lr": train_config.lr,
            "batch_size": train_config.batch_size
        })

        # 3. Training Loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = eval_one_epoch(model, val_loader, criterion, device)

            print(f"[Epoch {epoch:02d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            # Early Stopping Check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
                print(f"   >>> New best model! Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   >>> No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"[Info] Early stopping at epoch {epoch}")
                    break

        # 4. Save & Test
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, f"{args.model}_best.pth")
            print(f"[Info] Model saved.")
            mlflow.log_artifact(f"{args.model}_best.pth")

        test_loss = eval_one_epoch(model, test_loader, criterion, device)
        print(f"[Test] MSE: {test_loss:.4f}")
        mlflow.log_metric("test_mse", test_loss)

if __name__ == "__main__":
    main()