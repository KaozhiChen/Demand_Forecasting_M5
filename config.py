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
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20

    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )


data_config = DataConfig()
train_config = TrainConfig()