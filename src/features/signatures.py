"""
Path Signature Features for Deep Hedging.

Implements time-augmented path signatures using pure PyTorch
(fallback when signatory is not available).

Signature features capture pathwise information:
- Order 1: increments
- Order 2: iterated integrals (area under curve)
- Order 3-5: higher-order path interactions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

try:
    import signatory
    HAS_SIGNATORY = True
except ImportError:
    HAS_SIGNATORY = False


class PureSignature:
    """
    Pure PyTorch implementation of path signatures.
    Used as fallback when signatory is not available.
    """
    
    @staticmethod
    def signature_depth_1(path: torch.Tensor) -> torch.Tensor:
        """Level 1 signature: path increments."""
        # path: (batch, length, channels)
        increments = path[:, -1, :] - path[:, 0, :]
        return increments
    
    @staticmethod
    def signature_depth_2(path: torch.Tensor) -> torch.Tensor:
        """Level 2 signature: includes iterated integrals."""
        batch, length, channels = path.shape
        
        # Level 1
        sig1 = path[:, -1, :] - path[:, 0, :]
        
        # Level 2: approximate iterated integrals
        # S^{ij} ≈ Σ_k (X^i_{k+1} - X^i_k) * X^j_k
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        sig2_list = []
        for i in range(channels):
            for j in range(channels):
                # Approximate integral
                integral = torch.sum(increments[:, :, i] * path[:, :-1, j], dim=1)
                sig2_list.append(integral)
        
        sig2 = torch.stack(sig2_list, dim=1)
        
        return torch.cat([sig1, sig2], dim=1)
    
    @staticmethod
    def signature_depth_3(path: torch.Tensor) -> torch.Tensor:
        """Level 3 signature (approximate)."""
        batch, length, channels = path.shape
        
        # Get level 1 and 2
        sig2 = PureSignature.signature_depth_2(path)
        
        # Level 3: higher order terms (simplified approximation)
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        sig3_list = []
        for i in range(channels):
            for j in range(channels):
                for k in range(channels):
                    # Approximate triple integral
                    integral = torch.sum(
                        increments[:, :, i] * path[:, :-1, j] * path[:, :-1, k], 
                        dim=1
                    ) / length
                    sig3_list.append(integral)
        
        sig3 = torch.stack(sig3_list, dim=1)
        
        return torch.cat([sig2, sig3], dim=1)


class SignatureTransform(nn.Module):
    """
    Compute path signatures for hedging.
    
    Args:
        depth: Signature truncation depth (order)
        augment_time: Whether to add time as first coordinate
        normalize: Whether to normalize signatures
    """
    
    def __init__(
        self,
        depth: int = 3,
        augment_time: bool = True,
        normalize: bool = True
    ):
        super().__init__()
        self.depth = depth
        self.augment_time = augment_time
        self.normalize = normalize
        self.use_signatory = HAS_SIGNATORY
    
    def time_augment(self, path: torch.Tensor) -> torch.Tensor:
        """Add time coordinate as first channel."""
        batch, length, channels = path.shape
        time = torch.linspace(0, 1, length, device=path.device)
        time = time.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
        return torch.cat([time, path], dim=-1)
    
    def compute_signature(self, path: torch.Tensor) -> torch.Tensor:
        """Compute signature of path."""
        if self.use_signatory:
            return signatory.signature(path, depth=self.depth)
        else:
            # Use pure PyTorch fallback
            if self.depth == 1:
                return PureSignature.signature_depth_1(path)
            elif self.depth == 2:
                return PureSignature.signature_depth_2(path)
            else:
                return PureSignature.signature_depth_3(path)
    
    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute signature features.
        
        Args:
            path: (batch, length, channels)
            
        Returns:
            signature: (batch, sig_dim)
        """
        if self.augment_time:
            path = self.time_augment(path)
        
        sig = self.compute_signature(path)
        
        if self.normalize:
            sig = sig / (torch.norm(sig, dim=1, keepdim=True) + 1e-8)
        
        return sig
    
    def get_signature_dim(self, n_channels: int) -> int:
        """Get output dimension of signature."""
        if self.augment_time:
            n_channels += 1
        
        # Signature dimension formula
        dim = 0
        for d in range(1, self.depth + 1):
            dim += n_channels ** d
        return dim


class WindowedSignature(nn.Module):
    """
    Compute signatures over sliding windows for sequential hedging.
    
    Useful for providing local path information to recurrent models.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        depth: int = 2,
        augment_time: bool = True
    ):
        super().__init__()
        self.window_size = window_size
        self.sig_transform = SignatureTransform(depth, augment_time)
    
    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute windowed signatures.
        
        Args:
            path: (batch, length, channels)
            
        Returns:
            windowed_sigs: (batch, length - window_size + 1, sig_dim)
        """
        batch, length, channels = path.shape
        
        signatures = []
        for t in range(self.window_size - 1, length):
            window = path[:, t - self.window_size + 1:t + 1, :]
            sig = self.sig_transform(window)
            signatures.append(sig)
        
        return torch.stack(signatures, dim=1)


class SignatureHedgingModel(nn.Module):
    """
    Hedging model using path signatures as input.
    
    Architecture:
    1. Compute signature of path up to current time
    2. Pass signature through feedforward network
    3. Output delta hedge
    """
    
    def __init__(
        self,
        n_channels: int,
        n_steps: int,
        hidden_dims: List[int] = [64, 32],
        sig_depth: int = 3,
        delta_scale: float = 1.5
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_steps = n_steps
        self.delta_scale = delta_scale
        
        self.sig_transform = SignatureTransform(depth=sig_depth, augment_time=True)
        sig_dim = self.sig_transform.get_signature_dim(n_channels)
        
        # Build feedforward network
        layers = []
        prev_dim = sig_dim + 2  # +2 for moneyness and tau
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute hedging deltas using signatures.
        
        Args:
            features: (batch, n_steps, n_features)
            
        Returns:
            deltas: (batch, n_steps)
        """
        batch, n_steps, n_features = features.shape
        
        deltas = []
        for t in range(n_steps):
            # Get path up to time t
            path = features[:, :t+1, :]
            
            if t == 0:
                # Not enough path for signature, use zeros
                sig = torch.zeros(batch, self.sig_transform.get_signature_dim(n_features), 
                                  device=features.device)
            else:
                sig = self.sig_transform(path)
            
            # Add current moneyness and tau (assume first 2 features)
            moneyness = features[:, t, 0:1]
            tau = features[:, t, 1:2]
            
            # Combine signature with current state
            combined = torch.cat([sig, moneyness, tau], dim=1)
            
            # Get delta
            raw_delta = self.network(combined)
            delta = self.delta_scale * torch.tanh(raw_delta)
            deltas.append(delta.squeeze(-1))
        
        return torch.stack(deltas, dim=1)


class SignatureConditionedLSTM(nn.Module):
    """
    LSTM model conditioned on path signatures.
    
    Uses signatures as additional context for LSTM predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        sig_depth: int = 2,
        sig_window: int = 5,
        delta_scale: float = 1.5
    ):
        super().__init__()
        self.delta_scale = delta_scale
        
        # Windowed signature for local path info
        self.windowed_sig = WindowedSignature(
            window_size=sig_window,
            depth=sig_depth,
            augment_time=True
        )
        sig_dim = self.windowed_sig.sig_transform.get_signature_dim(input_dim)
        
        # LSTM with signature-augmented input
        self.lstm = nn.LSTM(
            input_size=input_dim + sig_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.sig_window = sig_window
        self.sig_dim = sig_dim
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with signature conditioning.
        
        Args:
            features: (batch, n_steps, input_dim)
            
        Returns:
            deltas: (batch, n_steps)
        """
        batch, n_steps, input_dim = features.shape
        
        # Compute windowed signatures
        # Pad beginning to get signatures for all timesteps
        padded = torch.cat([
            torch.zeros(batch, self.sig_window - 1, input_dim, device=features.device),
            features
        ], dim=1)
        
        sigs = self.windowed_sig(padded)  # (batch, n_steps, sig_dim)
        
        # Concatenate features with signatures
        combined = torch.cat([features, sigs], dim=-1)
        
        # LSTM forward
        output, _ = self.lstm(combined)
        
        # Get deltas
        raw_deltas = self.fc(output).squeeze(-1)
        deltas = self.delta_scale * torch.tanh(raw_deltas)
        
        return deltas
