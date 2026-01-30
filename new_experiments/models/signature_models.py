"""
Signature-Based Models for Deep Hedging

Paper-faithful implementation:
- Time-augmented path signatures
- Rolling window computation
- Normalized features
- Both LSTM and MLP variants
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from .base_model import BaseHedgingModel


class SignatureComputer(nn.Module):
    """
    Compute truncated path signatures.
    
    Given time-augmented path X_t = (t, S_t, log S_t, vol_t),
    computes signatures up to order m.
    """
    
    def __init__(
        self,
        input_dim: int,
        sig_order: int = 3,
        window_size: int = 5,
        normalize: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sig_order = sig_order
        self.window_size = window_size
        self.normalize = normalize
        
        # Compute signature dimension for each level
        d = input_dim + 1  # +1 for time augmentation
        self.level_dims = []
        for m in range(1, sig_order + 1):
            self.level_dims.append(d ** m)
        
        self.total_dim = sum(self.level_dims)
        self.augmented_dim = d
        
        # Normalization (learnable)
        if normalize:
            self.norm = nn.LayerNorm(self.total_dim)
    
    def _time_augment(self, path: torch.Tensor) -> torch.Tensor:
        """Add time coordinate as first channel."""
        batch, length, channels = path.shape
        time = torch.linspace(0, 1, length, device=path.device)
        time = time.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
        return torch.cat([time, path], dim=-1)
    
    def _compute_level1(self, path: torch.Tensor) -> torch.Tensor:
        """Level-1: path increments."""
        return path[:, -1, :] - path[:, 0, :]
    
    def _compute_level2(self, path: torch.Tensor) -> torch.Tensor:
        """Level-2: iterated integrals (area terms)."""
        batch, length, d = path.shape
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        sig2 = []
        for i in range(d):
            for j in range(d):
                # S^{ij} = Σ_k dX^i_k * X^j_{k-1}
                integral = torch.sum(increments[:, :, i] * path[:, :-1, j], dim=1)
                sig2.append(integral)
        
        return torch.stack(sig2, dim=1)
    
    def _compute_level3(self, path: torch.Tensor) -> torch.Tensor:
        """Level-3: higher-order terms."""
        batch, length, d = path.shape
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        sig3 = []
        scale = 1.0 / max(1, length - 1)
        
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    integral = torch.sum(
                        increments[:, :, i] * path[:, :-1, j] * path[:, :-1, k],
                        dim=1
                    ) * scale
                    sig3.append(integral)
        
        return torch.stack(sig3, dim=1)
    
    def _compute_level4(self, path: torch.Tensor) -> torch.Tensor:
        """Level-4: fourth-order terms (approximated)."""
        batch, length, d = path.shape
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        # Subsample for efficiency
        sig4 = []
        scale = 1.0 / max(1, (length - 1) ** 2)
        
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        integral = torch.sum(
                            increments[:, :, i] * path[:, :-1, j] * 
                            path[:, :-1, k] * path[:, :-1, l],
                            dim=1
                        ) * scale
                        sig4.append(integral)
        
        return torch.stack(sig4, dim=1)
    
    def compute_signature(self, path: torch.Tensor) -> torch.Tensor:
        """Compute full truncated signature."""
        levels = [self._compute_level1(path)]
        
        if self.sig_order >= 2:
            levels.append(self._compute_level2(path))
        
        if self.sig_order >= 3:
            levels.append(self._compute_level3(path))
        
        if self.sig_order >= 4:
            levels.append(self._compute_level4(path))
        
        sig = torch.cat(levels, dim=1)
        
        if self.normalize:
            sig = self.norm(sig)
        
        return sig
    
    def forward(self, features: torch.Tensor, t: int) -> torch.Tensor:
        """
        Compute rolling signature at timestep t.
        
        Args:
            features: (batch, n_steps, input_dim)
            t: current timestep
            
        Returns:
            signature: (batch, total_dim)
        """
        batch = features.size(0)
        
        # Get window
        start = max(0, t - self.window_size + 1)
        window = features[:, start:t+1, :]
        
        # Pad if needed
        if window.size(1) < self.window_size:
            pad_size = self.window_size - window.size(1)
            padding = window[:, 0:1, :].expand(-1, pad_size, -1)
            window = torch.cat([padding, window], dim=1)
        
        # Time augment
        window = self._time_augment(window)
        
        return self.compute_signature(window)


class SignatureLSTM(BaseHedgingModel):
    """
    Signature-conditioned LSTM hedger.
    
    Input at each step: [Sig^m_k, state_k]
    Architecture: LSTM with signature features concatenated
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        sig_order: int = 3,
        sig_window: int = 5,
        fc_layers: int = 2,
        dropout: float = 0.1,
        delta_max: float = 1.5,
        normalize_sig: bool = True
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sig_order = sig_order
        
        # Signature computer
        self.sig_computer = SignatureComputer(
            input_dim=input_dim,
            sig_order=sig_order,
            window_size=sig_window,
            normalize=normalize_sig
        )
        
        # Combined input dimension
        combined_dim = input_dim + self.sig_computer.total_dim
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output head
        fc_modules = []
        in_dim = hidden_size
        for i in range(fc_layers - 1):
            fc_modules.extend([
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_size
        fc_modules.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*fc_modules)
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with signature conditioning."""
        batch, n_steps, _ = features.shape
        device = features.device
        
        # Compute signatures for all timesteps
        sig_features = []
        for t in range(n_steps):
            sig = self.sig_computer(features, t)
            sig_features.append(sig)
        sig_features = torch.stack(sig_features, dim=1)
        
        # Concatenate with original features
        combined = torch.cat([features, sig_features], dim=-1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(combined)
        raw_deltas = self.fc(lstm_out).squeeze(-1)
        
        return self.bound_delta(raw_deltas)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sig_order': self.sig_order,
            'sig_dim': self.sig_computer.total_dim
        })
        return config


class SignatureMLP(BaseHedgingModel):
    """
    Non-recurrent signature MLP hedger.
    
    Tests whether recurrence is necessary for hedging.
    Uses rolling signatures without temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        n_layers: int = 3,
        sig_order: int = 3,
        sig_window: int = 5,
        dropout: float = 0.1,
        delta_max: float = 1.5,
        normalize_sig: bool = True
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        
        # Signature computer
        self.sig_computer = SignatureComputer(
            input_dim=input_dim,
            sig_order=sig_order,
            window_size=sig_window,
            normalize=normalize_sig
        )
        
        # MLP input: signature + current state
        mlp_input_dim = self.sig_computer.total_dim + input_dim
        
        # Build MLP
        layers = []
        in_dim = mlp_input_dim
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas using signature + MLP (no recurrence)."""
        batch, n_steps, _ = features.shape
        
        deltas = []
        for t in range(n_steps):
            sig = self.sig_computer(features, t)
            state = features[:, t, :]
            combined = torch.cat([sig, state], dim=-1)
            raw_delta = self.mlp(combined)
            deltas.append(raw_delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)


class SigFormerBlock(nn.Module):
    """
    SigFormer-style attention over signature levels.
    
    Applies separate attention to each signature level,
    then combines for final representation.
    """
    
    def __init__(
        self,
        level_dims: List[int],
        embed_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_levels = len(level_dims)
        self.embed_dim = embed_dim
        
        # Embedding for each level
        self.level_embeddings = nn.ModuleList([
            nn.Linear(dim, embed_dim) for dim in level_dims
        ])
        
        # Attention per level
        self.level_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(self.n_levels)
        ])
        
        # Combination
        self.combiner = nn.Sequential(
            nn.Linear(embed_dim * self.n_levels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, sig_levels: List[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
        """
        Attend over signature levels.
        
        Args:
            sig_levels: list of (batch, level_dim) tensors
            query: (batch, embed_dim) query vector
            
        Returns:
            attended: (batch, embed_dim)
        """
        attended_levels = []
        
        for i, (level, embed, attn) in enumerate(zip(
            sig_levels, self.level_embeddings, self.level_attention
        )):
            # Embed level
            embedded = embed(level).unsqueeze(1)  # (batch, 1, embed_dim)
            
            # Self-attention (query attends to level)
            q = query.unsqueeze(1)
            attn_out, _ = attn(q, embedded, embedded)
            attended_levels.append(attn_out.squeeze(1))
        
        # Combine all levels
        combined = torch.cat(attended_levels, dim=-1)
        return self.combiner(combined)


class SigFormerHedger(BaseHedgingModel):
    """
    SigFormer-style hedger with level-wise attention.
    
    Implements the idea of attending to different signature levels
    separately, allowing the model to weight geometric properties.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        sig_order: int = 3,
        sig_window: int = 5,
        dropout: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        
        # Signature computer
        self.sig_computer = SignatureComputer(
            input_dim=input_dim,
            sig_order=sig_order,
            window_size=sig_window,
            normalize=False  # Normalize per-level instead
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # SigFormer block
        self.sigformer = SigFormerBlock(
            level_dims=self.sig_computer.level_dims,
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        self.apply(self.init_weights)
    
    def _split_signature_levels(self, sig: torch.Tensor) -> List[torch.Tensor]:
        """Split concatenated signature into levels."""
        levels = []
        start = 0
        for dim in self.sig_computer.level_dims:
            levels.append(sig[:, start:start+dim])
            start += dim
        return levels
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with SigFormer attention."""
        batch, n_steps, _ = features.shape
        
        deltas = []
        for t in range(n_steps):
            # Compute signature
            sig = self.sig_computer(features, t)
            sig_levels = self._split_signature_levels(sig)
            
            # Encode current state
            state = features[:, t, :]
            state_encoded = self.state_encoder(state)
            
            # Attend over signature levels
            attended = self.sigformer(sig_levels, state_encoded)
            
            # Combine and output
            combined = torch.cat([state_encoded, attended], dim=-1)
            raw_delta = self.output_head(combined)
            deltas.append(raw_delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)
