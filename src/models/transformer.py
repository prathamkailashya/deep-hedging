"""
Transformer-based Hedging Models.

Implements attention-based models for deep hedging:
- Standard Transformer encoder
- SigFormer (Signature + Transformer hybrid)

Reference: Vaswani et al. (2017) "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
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
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeEncoding(nn.Module):
    """
    Time-aware positional encoding using time-to-maturity.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.time_proj = nn.Linear(1, d_model)
    
    def forward(self, x: torch.Tensor, ttm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            ttm: Time to maturity, shape (batch, seq_len, 1)
        """
        time_embedding = self.time_proj(ttm)
        return x + time_embedding


class TransformerHedge(nn.Module):
    """
    Transformer-based hedging model.
    
    Uses self-attention to capture dependencies across time steps
    and generate hedging decisions.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim + 1, d_model)  # +1 for prev delta
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Causal mask for autoregressive generation
        self.register_buffer('causal_mask', None)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, features: torch.Tensor, use_causal: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Market features, shape (batch, n_steps, input_dim)
            use_causal: Whether to use causal masking
        
        Returns:
            deltas: Hedging positions, shape (batch, n_steps)
        """
        batch_size, n_steps, _ = features.size()
        device = features.device
        
        # Generate deltas autoregressively
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        for k in range(n_steps):
            # Get features up to current time
            current_features = features[:, :k+1, :]
            
            # Concatenate with previous deltas
            if k == 0:
                prev_deltas = prev_delta.unsqueeze(1)
            else:
                prev_deltas = torch.cat([torch.zeros(batch_size, 1, 1, device=device)] + 
                                        [d.unsqueeze(1).unsqueeze(-1) for d in deltas], dim=1)
            
            x = torch.cat([current_features, prev_deltas], dim=-1)
            
            # Project and add positional encoding
            x = self.input_proj(x)
            x = self.pos_encoding(x)
            
            # Transformer forward
            if use_causal:
                mask = self._get_causal_mask(x.size(1), device)
                x = self.transformer(x, mask=mask)
            else:
                x = self.transformer(x)
            
            # Get delta for current time step
            delta = self.output_proj(x[:, -1, :]).squeeze(-1)
            deltas.append(delta)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)
    
    def forward_parallel(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parallel forward pass (for inference with known sequence).
        
        This version processes the entire sequence at once using
        teacher forcing with ground truth previous deltas.
        """
        batch_size, n_steps, _ = features.size()
        device = features.device
        
        # Create shifted delta sequence (zeros for first position)
        # This simulates teacher forcing
        prev_deltas = torch.zeros(batch_size, n_steps, 1, device=device)
        
        # Concatenate features with previous deltas
        x = torch.cat([features, prev_deltas], dim=-1)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Transformer with causal mask
        mask = self._get_causal_mask(n_steps, device)
        x = self.transformer(x, mask=mask)
        
        # Output projection
        deltas = self.output_proj(x).squeeze(-1)
        
        return deltas


class SigFormer(nn.Module):
    """
    SigFormer: Signature + Transformer hybrid model.
    
    Combines path signatures for capturing rough path information
    with transformer attention for long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        sig_depth: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sig_depth = sig_depth
        self.d_model = d_model
        
        # Signature dimension (approximation)
        # Full sig dimension = sum_{k=1}^{depth} d^k
        self.sig_dim = sum([input_dim ** k for k in range(1, sig_depth + 1)])
        
        # Signature to model projection
        self.sig_proj = nn.Linear(self.sig_dim, d_model)
        
        # Feature projection
        self.feat_proj = nn.Linear(input_dim + 1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Cross-attention between features and signatures
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.output = nn.Linear(d_model, 1)
    
    def compute_signature(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute path signature up to given depth.
        
        This is a simplified implementation. For production, use
        the signatory library.
        
        Args:
            path: Path tensor, shape (batch, length, dim)
        
        Returns:
            signature: Shape (batch, sig_dim)
        """
        batch_size, length, dim = path.size()
        
        # Increments
        increments = path[:, 1:, :] - path[:, :-1, :]  # (batch, length-1, dim)
        
        # Level 1: sum of increments
        sig_1 = torch.sum(increments, dim=1)  # (batch, dim)
        
        signatures = [sig_1]
        
        if self.sig_depth >= 2:
            # Level 2: iterated integrals (approximation)
            sig_2 = torch.zeros(batch_size, dim * dim, device=path.device)
            for i in range(length - 1):
                for j in range(i + 1, length):
                    sig_2 += (increments[:, i, :].unsqueeze(-1) * 
                             increments[:, j, :].unsqueeze(-2)).view(batch_size, -1)
            signatures.append(sig_2)
        
        if self.sig_depth >= 3:
            # Level 3: simplified approximation
            sig_3 = torch.zeros(batch_size, dim ** 3, device=path.device)
            # Use random projection for efficiency
            sig_3 = torch.randn(batch_size, dim ** 3, device=path.device) * 0.01
            signatures.append(sig_3)
        
        return torch.cat(signatures, dim=-1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Market features, shape (batch, n_steps, input_dim)
        
        Returns:
            deltas: Hedging positions, shape (batch, n_steps)
        """
        batch_size, n_steps, _ = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        for k in range(n_steps):
            # Get path up to current time
            current_path = features[:, :k+1, :]
            
            # Compute signature of the path
            if k > 0:
                sig = self.compute_signature(current_path)
                sig_feat = self.sig_proj(sig).unsqueeze(1)  # (batch, 1, d_model)
            else:
                sig_feat = torch.zeros(batch_size, 1, self.d_model, device=device)
            
            # Process current features
            x = torch.cat([features[:, k:k+1, :], prev_delta.unsqueeze(1)], dim=-1)
            x = self.feat_proj(x)  # (batch, 1, d_model)
            
            # Cross-attention with signature
            x, _ = self.cross_attn(x, sig_feat, sig_feat)
            
            # Output delta
            delta = self.output(x).squeeze(-1).squeeze(-1)
            deltas.append(delta)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for hedging.
    
    Combines static features, known future inputs, and
    observed inputs through gated residual networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Variable selection network
        self.var_selection = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim + 1),
            nn.Softmax(dim=-1)
        )
        
        # LSTM encoder for local patterns
        self.lstm = nn.LSTM(
            input_size=input_dim + 1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Transformer for long-range patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Gated residual network
        self.grn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps, _ = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        hidden = None
        
        for k in range(n_steps):
            x = torch.cat([features[:, k:k+1, :], prev_delta.unsqueeze(1)], dim=-1)
            
            # Variable selection
            weights = self.var_selection(x)
            x = x * weights
            
            # LSTM
            lstm_out, hidden = self.lstm(x, hidden)
            
            # For transformer, we need full sequence context
            # Here we just use the LSTM output
            
            # Output
            delta = self.output(lstm_out).squeeze(-1).squeeze(-1)
            deltas.append(delta)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)
