# models/lstm.py
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Simple LSTM baseline for next-day forecasting.
    Input shape:  (batch, seq_len, input_dim)
    Output shape: (batch, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch rules
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # lstm_out: (batch, seq_len, hidden_dim)

        last_hidden = lstm_out[:, -1, :]   # (batch, hidden_dim)

        y_hat = self.fc(last_hidden)       # (batch, 1)

        return y_hat