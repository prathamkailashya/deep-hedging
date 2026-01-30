"""
Transformer and SigFormer-style Models for Deep Hedging.

Implements:
- Encoder-only Transformer for hedging
- SigFormer: Transformer with signature token embeddings
- Positional encoding for time
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time steps.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeEncoding(nn.Module):
    """
    Learnable time encoding based on time-to-maturity.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
    
    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: (batch, seq_len, 1) time-to-maturity
        Returns:
            encoding: (batch, seq_len, d_model)
        """
        return self.time_embed(tau)


class TransformerHedgingModel(nn.Module):
    """
    Encoder-only Transformer for hedging.
    
    Uses causal attention to ensure delta at time t only depends on info up to t.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        delta_scale: float = 1.5,
        max_len: int = 100
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.d_model = d_model
        
        # Input embedding
        self.input_embed = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: (batch, seq_len, input_dim)
            
        Returns:
            deltas: (batch, seq_len)
        """
        batch, seq_len, _ = features.shape
        
        # Embed input
        x = self.input_embed(features)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask
        mask = self._generate_causal_mask(seq_len, features.device)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)
        
        # Project to delta
        raw_deltas = self.output_proj(encoded).squeeze(-1)
        deltas = self.delta_scale * torch.tanh(raw_deltas)
        
        return deltas


class SigFormerModel(nn.Module):
    """
    SigFormer-style model: Transformer with signature token embeddings.
    
    Uses windowed signatures as additional tokens/features for the Transformer.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        sig_depth: int = 2,
        sig_window: int = 5,
        delta_scale: float = 1.5
    ):
        super().__init__()
        self.delta_scale = delta_scale
        self.sig_window = sig_window
        
        # Signature computation (import here to avoid circular imports)
        from features.signatures import WindowedSignature
        self.windowed_sig = WindowedSignature(
            window_size=sig_window,
            depth=sig_depth,
            augment_time=True
        )
        sig_dim = self.windowed_sig.sig_transform.get_signature_dim(input_dim)
        
        # Combined input embedding (features + signature)
        self.input_embed = nn.Linear(input_dim + sig_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, 100, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.sig_dim = sig_dim
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with signature augmentation.
        
        Args:
            features: (batch, seq_len, input_dim)
            
        Returns:
            deltas: (batch, seq_len)
        """
        batch, seq_len, input_dim = features.shape
        
        # Compute windowed signatures
        padded = torch.cat([
            torch.zeros(batch, self.sig_window - 1, input_dim, device=features.device),
            features
        ], dim=1)
        sigs = self.windowed_sig(padded)  # (batch, seq_len, sig_dim)
        
        # Concatenate features with signatures
        combined = torch.cat([features, sigs], dim=-1)
        
        # Embed
        x = self.input_embed(combined)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Generate causal mask
        mask = self._generate_causal_mask(seq_len, features.device)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)
        
        # Project to delta
        raw_deltas = self.output_proj(encoded).squeeze(-1)
        deltas = self.delta_scale * torch.tanh(raw_deltas)
        
        return deltas


class CrossAttentionHedging(nn.Module):
    """
    Cross-attention model where hedging decisions attend to the full price history.
    
    Uses separate query (current state) and key-value (price history) streams.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        delta_scale: float = 1.5
    ):
        super().__init__()
        self.delta_scale = delta_scale
        
        # Embed price history (key-value)
        self.history_embed = nn.Linear(input_dim, d_model)
        
        # Embed current state (query)
        self.query_embed = nn.Linear(input_dim, d_model)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Output
        self.output_proj = nn.Linear(d_model, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention.
        
        Args:
            features: (batch, seq_len, input_dim)
            
        Returns:
            deltas: (batch, seq_len)
        """
        batch, seq_len, _ = features.shape
        
        # Embed history
        history = self.history_embed(features)  # (batch, seq_len, d_model)
        
        deltas = []
        for t in range(seq_len):
            # Query: current state
            query = self.query_embed(features[:, t:t+1, :])  # (batch, 1, d_model)
            
            # Key-Value: history up to t
            kv = history[:, :t+1, :]  # (batch, t+1, d_model)
            
            # Apply cross-attention layers
            x = query
            for attn, ln in zip(self.cross_attn_layers, self.layer_norms):
                attn_out, _ = attn(x, kv, kv)
                x = ln(x + attn_out)
            
            # Project to delta
            raw_delta = self.output_proj(x).squeeze(-1).squeeze(-1)
            delta = self.delta_scale * torch.tanh(raw_delta)
            deltas.append(delta)
        
        return torch.stack(deltas, dim=1)
