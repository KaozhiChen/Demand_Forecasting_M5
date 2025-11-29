# models/transformer.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class Transformer(nn.Module):
    """
    Standard Transformer for time series forecasting.
    Uses sinusoidal positional encoding.
    
    Input shape:  (batch, seq_len, input_dim)
    Output shape: (batch, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.transformer_encoder(x)
        
        # Use the last time step's output: (batch, seq_len, d_model) -> (batch, d_model)
        x = x[:, -1, :]
        
        # Output projection: (batch, d_model) -> (batch, 1)
        y_hat = self.output_projection(x)
        
        return y_hat
