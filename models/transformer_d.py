# models/transformer_d.py
import torch
import torch.nn as nn


class TransformerD(nn.Module):
    """
    Transformer-D: Transformer with Date-based positional encoding.
    
    Key differences from standard Transformer:
    - Uses date features (day_of_year, doy_sin, doy_cos) as input features
    - Removes standard sinusoidal positional encoding layer
    - Time position information is explicitly encoded in input features
    
    Input shape:  (batch, seq_len, input_dim) where input_dim includes date features
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
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # NO positional encoding layer - time info comes from input features
        
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
        Note: input_dim should include date_pe_features (day_of_year, doy_sin, doy_cos)
        """
        # Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # NO positional encoding - time info is already in input features
        
        # Transformer encoder: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.transformer_encoder(x)
        
        # Use the last time step's output: (batch, seq_len, d_model) -> (batch, d_model)
        x = x[:, -1, :]
        
        # Output projection: (batch, d_model) -> (batch, 1)
        y_hat = self.output_projection(x)
        
        return y_hat

