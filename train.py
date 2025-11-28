# train.py
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from config import data_config, train_config
from data import load_ca1_features, create_dataloaders
from models.lstm import LSTM



# Model selection (extend here when adding Transformer models)
def get_model(model_name: str, input_dim: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "lstm":
        model = LSTM(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

# Training & Validation Loop
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
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        help="Model name: lstm (later: transformer_s, transformer_d, ...)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    # Early stopping configuration from config
    patience = train_config.early_stop_patience
    min_delta = train_config.early_stop_min_delta

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Device selection (from config)
    device_str = train_config.device
    device = torch.device(device_str)
    print(f"[Info] Using device: {device}")
    print(f"[Info] Random seed: {args.seed}")

    # 1. Load data
    print("[Info] Loading data...")
    df_all = load_ca1_features()

    # For LSTM / baseline Transformer, use simple features first
    feature_cols = data_config.simple_features
    input_dim = len(feature_cols)

    train_loader, val_loader, test_loader = create_dataloaders(
        df_all,
        feature_cols=feature_cols,
        batch_size=train_config.batch_size,
        seq_len=data_config.seq_len,
        stride=data_config.stride,
    )

    # 2. Initialize model
    print(f"[Info] Building model: {args.model}")
    model = get_model(args.model, input_dim=input_dim).to(device)
    print(f"[Info] Model moved to {device}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=train_config.lr)
    print("[Info] Starting training...")

    # Initialize MLflow
    mlflow.set_experiment("M5_Demand_Forecasting")
    
    with mlflow.start_run(run_name=f"{args.model}_{args.epochs}epochs"):
        # Log hyperparameters
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.lr,
            "seq_len": data_config.seq_len,
            "stride": data_config.stride,
            "input_dim": input_dim,
            "device": str(device),
            "early_stop": True,
            "patience": patience,
            "min_delta": min_delta,
        })
        
        # Log model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0  # Counter for early stopping

        # 3. Training loop
        for epoch in range(1, args.epochs + 1):
            print(f"[Epoch {epoch:02d}/{args.epochs}] Training...")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"[Epoch {epoch:02d}/{args.epochs}] Validating...")
            val_loss = eval_one_epoch(model, val_loader, criterion, device)

            print(
                f"[Epoch {epoch:02d}] "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            # Log metrics for each epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, step=epoch)

            # Early stopping logic (enabled by default for all models)
            # Check if validation loss improved
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0  # Reset counter on improvement
                print(f"[Info] Validation loss improved to {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"[Info] No improvement for {patience_counter}/{patience} epochs")
                
                # Early stopping trigger
                if patience_counter >= patience:
                    print(f"[Info] Early stopping triggered after {epoch} epochs")
                    print(f"[Info] Best validation loss: {best_val_loss:.4f}")
                    mlflow.log_param("early_stopped", True)
                    mlflow.log_param("stopped_at_epoch", epoch)
                    break

        # 4. Save best model checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
            checkpoint_path = f"{args.model}_best.pth"
            torch.save(best_state, checkpoint_path)
            print(f"[Info] Best model saved to {checkpoint_path}")
            
            # Log best validation loss
            mlflow.log_metric("best_val_loss", best_val_loss)
            
            # Save model to MLflow
            mlflow.pytorch.log_model(model, "model")
            
            # Also log the checkpoint file as artifact
            mlflow.log_artifact(checkpoint_path)

        # 5. Test set evaluation (using best model)
        test_loss = eval_one_epoch(model, test_loader, criterion, device)
        print(f"[Test] MSE={test_loss:.4f}")
        
        # Log test metrics
        mlflow.log_metrics({
            "test_mse": test_loss,
            "test_rmse": test_loss ** 0.5,  # RMSE = sqrt(MSE)
        })
        
        print(f"[Info] MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()