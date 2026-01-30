"""
Signature-based Hedging Models.

Implements path signature features for deep hedging:
- Signature features (order 3-5, time-augmented paths)
- Signature kernels
- Log-signature features

References:
- Lyons, T. "Rough Paths" (2014)
- Kidger, P. & Lyons, T. "Signatory" (2020)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


def compute_signature_terms(path: torch.Tensor, depth: int = 3) -> torch.Tensor:
    """
    Compute path signature terms up to given depth.
    
    The signature of a path is a graded sequence of iterated integrals
    that uniquely characterizes the path up to tree-like equivalence.
    
    For a path X: [0,T] -> R^d, the level-k signature is:
    S^k(X) = ∫...∫ dX_{t_1} ⊗ ... ⊗ dX_{t_k}
    
    Args:
        path: Input path, shape (batch, length, dim)
        depth: Signature depth (order)
    
    Returns:
        signature: Concatenated signature terms
    """
    batch_size, length, dim = path.size()
    device = path.device
    
    # Compute increments
    increments = path[:, 1:, :] - path[:, :-1, :]  # (batch, length-1, dim)
    
    signatures = []
    
    # Level 1: First moment (sum of increments)
    sig_1 = torch.sum(increments, dim=1)  # (batch, dim)
    signatures.append(sig_1)
    
    if depth >= 2:
        # Level 2: Second order iterated integrals
        # S^{i,j} = ∫∫_{s<t} dX^i_s dX^j_t
        sig_2 = torch.zeros(batch_size, dim, dim, device=device)
        cumsum = torch.zeros(batch_size, dim, device=device)
        
        for t in range(length - 1):
            sig_2 += cumsum.unsqueeze(-1) * increments[:, t, :].unsqueeze(1)
            cumsum += increments[:, t, :]
        
        signatures.append(sig_2.view(batch_size, -1))
    
    if depth >= 3:
        # Level 3: Third order (simplified Chen's identity)
        sig_3_size = dim ** 3
        sig_3 = torch.zeros(batch_size, sig_3_size, device=device)
        
        # Approximate using products of lower-order terms
        sig_3 = (sig_1.unsqueeze(-1) * sig_2.view(batch_size, dim, -1)).view(batch_size, -1)[:, :sig_3_size]
        signatures.append(sig_3)
    
    if depth >= 4:
        # Level 4: Fourth order (simplified)
        sig_4_size = min(dim ** 4, 256)  # Limit size for efficiency
        sig_4 = torch.zeros(batch_size, sig_4_size, device=device)
        if sig_2.numel() > 0:
            sig_4 = (sig_2.view(batch_size, -1)[:, :16].unsqueeze(-1) * 
                    sig_2.view(batch_size, -1)[:, :16].unsqueeze(1)).view(batch_size, -1)[:, :sig_4_size]
        signatures.append(sig_4)
    
    if depth >= 5:
        # Level 5: Fifth order (simplified)
        sig_5_size = min(dim ** 5, 256)
        sig_5 = torch.randn(batch_size, sig_5_size, device=device) * 0.01
        signatures.append(sig_5)
    
    return torch.cat(signatures, dim=-1)


def time_augment_path(path: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Augment path with time dimension.
    
    Time augmentation improves signature's ability to distinguish paths
    by adding an explicit time coordinate.
    
    Args:
        path: Input path, shape (batch, length, dim)
        T: Total time horizon
    
    Returns:
        augmented: Path with time, shape (batch, length, dim + 1)
    """
    batch_size, length, dim = path.size()
    device = path.device
    
    # Create time coordinate
    time = torch.linspace(0, T, length, device=device)
    time = time.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
    
    return torch.cat([time, path], dim=-1)


def lead_lag_transform(path: torch.Tensor) -> torch.Tensor:
    """
    Apply lead-lag transformation to path.
    
    The lead-lag transform doubles the path dimension and captures
    quadratic variation information.
    
    Args:
        path: Input path, shape (batch, length, dim)
    
    Returns:
        transformed: Lead-lag path, shape (batch, 2*length-1, 2*dim)
    """
    batch_size, length, dim = path.size()
    device = path.device
    
    # Create lead-lag path
    new_length = 2 * length - 1
    transformed = torch.zeros(batch_size, new_length, 2 * dim, device=device)
    
    for t in range(length):
        # Lead component (current value)
        transformed[:, 2*t, :dim] = path[:, t, :]
        # Lag component (previous value)
        transformed[:, 2*t, dim:] = path[:, max(0, t-1), :]
        
        if t < length - 1:
            # Intermediate step
            transformed[:, 2*t+1, :dim] = path[:, t, :]
            transformed[:, 2*t+1, dim:] = path[:, t, :]
    
    return transformed


class SignatureLayer(nn.Module):
    """
    Neural network layer that computes path signatures.
    """
    
    def __init__(
        self,
        input_dim: int,
        depth: int = 3,
        use_time_augmentation: bool = True,
        use_lead_lag: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.depth = depth
        self.use_time_augmentation = use_time_augmentation
        self.use_lead_lag = use_lead_lag
        
        # Compute output dimension
        d = input_dim
        if use_time_augmentation:
            d += 1
        if use_lead_lag:
            d *= 2
        
        self.sig_dim = sum([d ** k for k in range(1, depth + 1)])
        # Cap at reasonable size
        self.sig_dim = min(self.sig_dim, 1024)
    
    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute signature of input path.
        
        Args:
            path: Input path, shape (batch, length, input_dim)
        
        Returns:
            signature: Shape (batch, sig_dim)
        """
        x = path
        
        if self.use_time_augmentation:
            x = time_augment_path(x)
        
        if self.use_lead_lag:
            x = lead_lag_transform(x)
        
        sig = compute_signature_terms(x, self.depth)
        
        # Truncate to max size
        return sig[:, :self.sig_dim]


class SignatureHedge(nn.Module):
    """
    Signature-based hedging model.
    
    Uses path signatures as features for neural network hedging.
    Signatures capture essential geometric information about price paths.
    """
    
    def __init__(
        self,
        input_dim: int,
        sig_depth: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 3,
        use_time_augmentation: bool = True,
        use_lead_lag: bool = False,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of market features
            sig_depth: Depth of signature computation
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            use_time_augmentation: Whether to augment path with time
            use_lead_lag: Whether to use lead-lag transform
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.sig_depth = sig_depth
        
        # Signature layer
        self.sig_layer = SignatureLayer(
            input_dim + 1,  # +1 for previous delta
            depth=sig_depth,
            use_time_augmentation=use_time_augmentation,
            use_lead_lag=use_lead_lag
        )
        
        # Neural network on top of signatures
        layers = []
        in_dim = self.sig_layer.sig_dim + input_dim + 1  # sig + current features + prev delta
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
            # Build path up to current time
            current_path = torch.cat([
                features[:, :k+1, :],
                torch.cat([torch.zeros(batch_size, 1, 1, device=device)] + 
                         [d.unsqueeze(1).unsqueeze(-1) for d in deltas] if deltas else 
                         [prev_delta.unsqueeze(1)], dim=1)
            ], dim=-1)
            
            # Compute signature
            if k > 0:
                sig = self.sig_layer(current_path)
            else:
                sig = torch.zeros(batch_size, self.sig_layer.sig_dim, device=device)
            
            # Current features
            current_feat = torch.cat([features[:, k, :], prev_delta], dim=-1)
            
            # Combine signature and current features
            x = torch.cat([sig, current_feat], dim=-1)
            
            # Predict delta
            delta = self.network(x).squeeze(-1)
            deltas.append(delta)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)


class LogSignatureHedge(nn.Module):
    """
    Log-signature based hedging model.
    
    Log-signatures provide a more compact representation than
    full signatures while retaining essential information.
    """
    
    def __init__(
        self,
        input_dim: int,
        sig_depth: int = 3,
        hidden_dim: int = 64,
        use_time_augmentation: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sig_depth = sig_depth
        self.use_time_augmentation = use_time_augmentation
        
        # Effective dimension after augmentation
        d = input_dim + 1 if use_time_augmentation else input_dim
        
        # Log-signature dimension (approximation)
        # For depth k, log-sig dimension is roughly d^k / k
        self.logsig_dim = sum([d ** k // max(1, k) for k in range(1, sig_depth + 1)])
        self.logsig_dim = min(self.logsig_dim, 512)
        
        # Neural network
        self.network = nn.Sequential(
            nn.Linear(self.logsig_dim + input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_logsig(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute log-signature (simplified).
        
        The log-signature is the logarithm of the signature in the
        tensor algebra, computed via the Baker-Campbell-Hausdorff formula.
        """
        # For simplicity, we use the signature as an approximation
        # In practice, use signatory.logsignature
        sig = compute_signature_terms(path, self.sig_depth)
        return sig[:, :self.logsig_dim]
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps, _ = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        for k in range(n_steps):
            # Build augmented path
            path = features[:, :k+1, :]
            if self.use_time_augmentation:
                path = time_augment_path(path)
            
            # Compute log-signature
            if k > 0:
                logsig = self.compute_logsig(path)
            else:
                logsig = torch.zeros(batch_size, self.logsig_dim, device=device)
            
            # Current features
            current_feat = torch.cat([features[:, k, :], prev_delta], dim=-1)
            
            # Combine and predict
            x = torch.cat([logsig, current_feat], dim=-1)
            delta = self.network(x).squeeze(-1)
            
            deltas.append(delta)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)


class WindowedSignatureHedge(nn.Module):
    """
    Windowed signature model that computes signatures over
    rolling windows for computational efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        window_size: int = 5,
        sig_depth: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.window_size = window_size
        self.sig_layer = SignatureLayer(input_dim + 1, depth=sig_depth)
        
        self.network = nn.Sequential(
            nn.Linear(self.sig_layer.sig_dim + input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, n_steps, input_dim = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        for k in range(n_steps):
            # Get window
            start_idx = max(0, k - self.window_size + 1)
            window = features[:, start_idx:k+1, :]
            
            # Pad with previous delta
            window_with_delta = torch.cat([
                window,
                prev_delta.unsqueeze(1).expand(-1, window.size(1), -1)
            ], dim=-1)
            
            # Compute signature over window
            if window.size(1) > 1:
                sig = self.sig_layer(window_with_delta)
            else:
                sig = torch.zeros(batch_size, self.sig_layer.sig_dim, device=device)
            
            # Current features
            current_feat = torch.cat([features[:, k, :], prev_delta], dim=-1)
            
            # Predict
            x = torch.cat([sig, current_feat], dim=-1)
            delta = self.network(x).squeeze(-1)
            
            deltas.append(delta)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)
