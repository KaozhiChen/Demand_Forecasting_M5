import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
        return self.dropout(x)


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
        max_seq_len: int = 500, 
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        # Baseline: (Continuous + Date Numbers)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (Standard Sinusoidal)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_seq_len)
        
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
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # 1. Input projection: (B, L, F) -> (B, L, D)
        x = self.input_projection(x)
        
        # 2. Add positional encoding & Dropout
        # Standard: Input -> Projection -> PE -> Dropout
        x = self.pos_encoder(x)
        
        # 3. Transformer encoder: (B, L, D) -> (B, L, D)
        x = self.transformer_encoder(x)
        
        # 4. Use the last time step's output
        x = x[:, -1, :]
        
        # 5. Output projection: (B, D) -> (B, 1)
        y_hat = self.output_projection(x)
        
        return y_hat