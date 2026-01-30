"""
Transformer Models for Deep Hedging

Financially correct implementation:
- Causal attention (no peeking)
- Positional encoding mandatory
- Encoder-only architecture
- Same bounded delta output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_model import BaseHedgingModel


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalTransformerEncoder(nn.Module):
    """
    Causal transformer encoder for time series.
    
    Uses causal masking to prevent looking into the future.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (upper triangular)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with causal masking."""
        seq_len = x.size(1)
        mask = self._generate_causal_mask(seq_len, x.device)
        return self.encoder(x, mask=mask)


class TransformerHedger(BaseHedgingModel):
    """
    Causal Transformer hedger.
    
    Architecture:
    - Input embedding
    - Positional encoding (mandatory)
    - Causal transformer encoder
    - Output projection to bounded delta
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        delta_max: float = 1.5,
        pos_encoding: str = 'sinusoidal'
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoder = LearnedPositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        self.transformer = CausalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate bounded deltas using causal transformer."""
        # Project input
        x = self.input_proj(features)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding with causal mask
        encoded = self.transformer(x)
        
        # Project to delta
        raw_deltas = self.output_proj(encoded).squeeze(-1)
        
        return self.bound_delta(raw_deltas)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'architecture': 'CausalTransformer'
        })
        return config


class TimeSeriesTransformer(BaseHedgingModel):
    """
    Enhanced transformer with time-aware features.
    
    Adds explicit time encoding and handles variable-length sequences.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # Feature embedding  
        self.feature_embed = nn.Linear(input_dim, d_model - d_model // 4)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer
        self.transformer = CausalTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with time-aware transformer."""
        batch, n_steps, _ = features.shape
        device = features.device
        
        # Create time features
        time = torch.linspace(0, 1, n_steps, device=device)
        time = time.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1)
        
        # Embed time and features separately
        time_emb = self.time_embed(time)
        feat_emb = self.feature_embed(features)
        
        # Combine
        x = torch.cat([feat_emb, time_emb], dim=-1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        encoded = self.transformer(x)
        
        # Output
        raw_deltas = self.output_proj(encoded).squeeze(-1)
        
        return self.bound_delta(raw_deltas)


class CrossAttentionTransformer(BaseHedgingModel):
    """
    Transformer with cross-attention between price and volatility streams.
    
    Separates price and volatility processing, then combines via cross-attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        # Assume input: [S_norm, log_moneyness, vol, tau]
        # Split into price features (0,1,3) and vol features (2,3)
        
        self.d_model = d_model
        
        # Separate embeddings
        self.price_embed = nn.Linear(3, d_model)  # S_norm, log_money, tau
        self.vol_embed = nn.Linear(2, d_model)    # vol, tau
        
        # Self-attention layers
        self.price_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.vol_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention: price attends to vol
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with cross-attention."""
        batch, n_steps, _ = features.shape
        
        # Split features
        price_feats = torch.stack([features[:,:,0], features[:,:,1], features[:,:,3]], dim=-1)
        vol_feats = torch.stack([features[:,:,2], features[:,:,3]], dim=-1)
        
        # Embed
        price_emb = self.price_embed(price_feats)
        vol_emb = self.vol_embed(vol_feats)
        
        # Causal mask
        mask = torch.triu(torch.ones(n_steps, n_steps, device=features.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Self-attention
        price_sa, _ = self.price_self_attn(price_emb, price_emb, price_emb, attn_mask=mask)
        price_sa = self.norm1(price_emb + price_sa)
        
        vol_sa, _ = self.vol_self_attn(vol_emb, vol_emb, vol_emb, attn_mask=mask)
        vol_sa = self.norm2(vol_emb + vol_sa)
        
        # Cross-attention: price queries attend to vol keys/values
        cross_out, _ = self.cross_attn(price_sa, vol_sa, vol_sa, attn_mask=mask)
        
        # Combine and output
        combined = torch.cat([price_sa, cross_out], dim=-1)
        raw_deltas = self.ffn(combined).squeeze(-1)
        
        return self.bound_delta(raw_deltas)
