# models/transformer_d.py
import torch
import torch.nn as nn
import math

# PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerD(nn.Module):
    """
    Transformer-D (Final Version):
    Hybrid Position Encoding = Standard Index PE + Semantic Date Embeddings.
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
        
        # Feature split: 5 continuous (Sales, Sin, Cos...), 2 categorical (Wday, Month)
        self.num_cont = 5
        self.num_cat = 2
        
        # 1. Projection layer
        self.cont_projection = nn.Linear(self.num_cont, d_model)
        
        # 2. Date Embeddings (capture periodic semantics)
        self.wday_emb = nn.Embedding(8, d_model)
        self.month_emb = nn.Embedding(13, d_model)
        
        # 3. Standard PE 
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=500)
        
        # 4. Transformer Backbone
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
        
        self.output_projection = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.05, 0.05)

    def forward(self, x):
        # 1. Manually split input
        x_cont = x[:, :, :self.num_cont]
        x_wday = x[:, :, 5].long()
        x_month = x[:, :, 6].long()

        # 2. Compute Embeddings
        feat_emb = self.cont_projection(x_cont)
        wday_emb = self.wday_emb(x_wday)
        month_emb = self.month_emb(x_month)
        
        # 3. Hybrid mechanism
        # Content (Feat) + Date Semantics (Wday/Month)
        x = feat_emb + wday_emb + month_emb
        
        # 4. Inject sequence order (Standard PE)
        x = self.pos_encoder(x)

        # 5. Encode and predict
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        y_hat = self.output_projection(x)
        
        return y_hat