import torch
import torch.nn as nn

class TransformerD(nn.Module):
    """
    Transformer-D: Date-Aware Transformer.
    
    Core Mechanisms:
    1. Hybrid Input Processing:
       - Continuous features (Sales, Sin/Cos, Events) -> Projected via Linear layer.
       - Categorical features (Wday, Month) -> Mapped via Embedding layers.
    2. Alternative Positional Encoding:
       - Removes standard Sinusoidal Positional Encoding.
       - Uses Wday and Month embeddings as semantic positional information
         added directly to the feature embeddings.
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
        
        # 1. Feature Split Definition (Based on config.py order)
        # config.all_features = continuous_features + categorical_features
        # Order: [sales, snap, holiday, sin, cos] + [wday, month]

        self.num_cont = 5  # First 5 columns are continuous
        self.num_cat = 2   # Last 2 columns are categorical (wday, month)
        
        # Safety check: Ensure input_dim matches our splitting logic
        if input_dim != self.num_cont + self.num_cat:
            raise ValueError(
                f"Input dim mismatch! Expected {self.num_cont + self.num_cat}, got {input_dim}. "
                "Check config.py feature lists."
            )

        # 2. Projection & Embedding Layers (The Core Innovation)
        
        # A. Continuous Feature Projection (5 -> d_model)
        self.cont_projection = nn.Linear(self.num_cont, d_model)
        
        # B. Date Feature Embeddings
        # wday: 1-7. Set size to 8 to accommodate index 7 (index 0 unused)
        self.wday_emb = nn.Embedding(num_embeddings=8, embedding_dim=d_model)
        
        # month: 1-12. Set size to 13 to accommodate index 12 (index 0 unused)
        self.month_emb = nn.Embedding(num_embeddings=13, embedding_dim=d_model)
        
        # 3. Transformer Backbone   
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
        
        # 4. Output Head
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights: Xavier for Linear, Uniform for Embedding."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.05, 0.05)

    def forward(self, x):
        """
        x shape: (batch, seq_len, 7)
        
        Manually slice x based on feature types:
        - x[:, :, 0:5] -> Continuous (sales, snap, holiday, sin, cos)
        - x[:, :, 5]   -> Wday (needs casting to long)
        - x[:, :, 6]   -> Month (needs casting to long)
        """
        # Step 1: Slice Inputs
        x_cont = x[:, :, :self.num_cont]        # (B, L, 5) float32
        
        # Must cast to long for Embedding lookup
        x_wday = x[:, :, 5].long()              # (B, L) int64
        x_month = x[:, :, 6].long()             # (B, L) int64

        # Step 2: Project & Embed 
        # Continuous projection
        feat_emb = self.cont_projection(x_cont) # (B, L, d_model)
        
        # Date Embeddings (Semantic position)
        wday_emb = self.wday_emb(x_wday)        # (B, L, d_model)
        month_emb = self.month_emb(x_month)     # (B, L, d_model)
        
        # Step 3: Combine (Addition)
        # Core Logic: Content + Date_Position
        # Here, wday and month embeddings ACT as the positional encoding.
        x = feat_emb + wday_emb + month_emb
        
        x = self.dropout(x)

        # Step 4: Transformer Encoding 
        # Note: No self.pos_encoder(x) here because we added date info above.
        x = self.transformer_encoder(x)
        
        # Step 5: Prediction
        # Use the output of the last time step
        x = x[:, -1, :]
        y_hat = self.output_projection(x)
        
        return y_hat