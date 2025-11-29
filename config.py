# config.py
from dataclasses import dataclass
from datetime import datetime
import torch

@dataclass
class DataConfig:
    # path to processed training data
    data_path: str = "./Data/training_data.csv"

    # sliding window configuration
    seq_len: int = 56
    stride: int = 1       

    # target
    target_col: str = "sales"

    # features used by ALL models
    simple_features: tuple = (
        "sales",
        "wday",
        "month",
        "year",
        "snap_CA",
        "is_holiday",
    )

    # date positional encoding (Transformer-D)
    date_pe_features: tuple = (
        "day_of_year",
        "doy_sin",
        "doy_cos",
    )

    # temporal split
    train_end: datetime = datetime(2014, 12, 31)
    val_end: datetime   = datetime(2015, 12, 31)


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 20
    
    # Early stopping configuration
    early_stop_patience: int = 3  
    early_stop_min_delta: float = 0.0  

    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Model architecture hyperparameters (used by Transformer models)
    # Standard Transformer and Transformer-D share these configs
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


data_config = DataConfig()
train_config = TrainConfig()