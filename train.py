# train.py
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam

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

    for x, y in loader:
        x = x.to(device)  # (B, L, F)
        y = y.to(device)  # (B, 1)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)

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
    args = parser.parse_args()

    # Device selection (from config)
    device_str = train_config.device
    device = torch.device(device_str)
    print(f"[Info] Using device: {device}")

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

    best_val_loss = float("inf")
    best_state = None

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

        # Simple best model selection
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # 4. Save best model checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint_path = f"{args.model}_best.pth"
        torch.save(best_state, checkpoint_path)
        print(f"[Info] Best model saved to {checkpoint_path}")

    # 5. Test set evaluation (using best model)
    test_loss = eval_one_epoch(model, test_loader, criterion, device)
    print(f"[Test] MSE={test_loss:.4f}")


if __name__ == "__main__":
    main()