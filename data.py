# data.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from config import data_config, train_config


# Load processed dataset
def load_ca1_features() -> pd.DataFrame:
    """Load the processed CA1 feature table."""
    path = data_config.data_path

    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = pd.read_parquet(path)

    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    print("[Data] Applying Log1p transformation to 'sales'...")
    df["sales"] = np.log1p(df["sales"])
    return df


# Temporal split (train/val/test)
def split_by_date(df: pd.DataFrame):
    train = df[df["date"] <= data_config.train_end]
    val = df[(df["date"] > data_config.train_end) &
             (df["date"] <= data_config.val_end)]
    test = df[df["date"] > data_config.val_end]
    return train, val, test


class ConstructDataset(Dataset):
    """
    Construct Dataset for time series forecasting:
    - Group by item_id
    - Apply sliding window for each item
    - Input:  past seq_len days of feature sequence
    - Target: next day's sales
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        seq_len: int,
        stride: int = 1,
        target_col: str = "sales",
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.feature_cols = list(feature_cols)
        self.target_col = target_col

        self.windows = []       # list[(item_id, start_idx)]
        self.group_arrays = {}  # item_id -> numpy array (N, F+1)

        # Group by item_id and preprocess each item's data
        for item_id, g in df.groupby("item_id"):
            g = g.sort_values("date").reset_index(drop=True)

            # Extract feature and target columns
            arr_x = g[self.feature_cols].to_numpy(dtype=np.float32)          # (N, F)
            arr_y = g[[self.target_col]].to_numpy(dtype=np.float32)          # (N, 1)

            # Concatenate into a (N, F+1) array, with target as the last column
            arr = np.concatenate([arr_x, arr_y], axis=1)
            n = len(arr)

            # Need at least seq_len + 1 days to create "past seq_len days → predict next day"
            max_start = n - seq_len - 1
            if max_start <= 0:
                continue

            self.group_arrays[item_id] = arr

            # Generate all window start positions
            for start in range(0, max_start + 1, stride):
                self.windows.append((item_id, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item_id, start = self.windows[idx]
        arr = self.group_arrays[item_id]

        end = start + self.seq_len
        window = arr[start:end]   # (L, F+1)
        next_row = arr[end]       # (F+1,)

        # First F columns are features, last column is target
        x = window[:, :-1]        # (L, F)
        y = next_row[-1]          # scalar

        x_tensor = torch.from_numpy(x)                         # (L, F)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(1)  # (1,)

        return x_tensor, y_tensor


def create_dataloaders(
    df_all: pd.DataFrame,
    feature_cols,
    batch_size: int | None = None,
    seq_len: int | None = None,
    stride: int | None = None,
):
    """
    From the complete dataframe:
    - Split by time into train/val/test
    - Build Dataset (with sliding window)
    - Wrap in DataLoader
    """
    if batch_size is None:
        batch_size = train_config.batch_size
    if seq_len is None:
        seq_len = data_config.seq_len
    if stride is None:
        stride = data_config.stride

    train_df, val_df, test_df = split_by_date(df_all)

    train_ds = ConstructDataset(
        train_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stride=stride,
        target_col=data_config.target_col,
    )
    val_ds = ConstructDataset(
        val_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stride=stride,
        target_col=data_config.target_col,
    )
    test_ds = ConstructDataset(
        test_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stride=stride,
        target_col=data_config.target_col,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    print(
        f"[Data] train samples: {len(train_ds)}, "
        f"val: {len(val_ds)}, test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader