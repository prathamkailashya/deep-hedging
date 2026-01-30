"""
Novel Hybrid Extensions for Deep Hedging

Implements:
1. Ensemble Hedger: median/mean/weighted combination
2. Regime-Aware Model: volatility-based policy switching
3. Distributionally Robust Hedger: trained across multiple models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from .base_model import BaseHedgingModel, LSTMHedger


class EnsembleHedger(BaseHedgingModel):
    """
    Ensemble of multiple hedging models.
    
    Combination methods:
    - median: robust to outliers
    - mean: simple average
    - weighted: learned weights
    - learned: neural network combiner
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        method: str = 'median',
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        self.models = nn.ModuleList(models)
        self.method = method
        self.n_models = len(models)
        
        if method == 'weighted':
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        elif method == 'learned':
            # Input: n_models deltas, output: 1 delta
            self.combiner = nn.Sequential(
                nn.Linear(self.n_models, self.n_models * 2),
                nn.ReLU(),
                nn.Linear(self.n_models * 2, 1)
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Combine model predictions."""
        # Get predictions from all models
        all_deltas = []
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                deltas = model(features)
            all_deltas.append(deltas)
        
        # Stack: (n_models, batch, n_steps)
        stacked = torch.stack(all_deltas, dim=0)
        
        if self.method == 'median':
            combined = torch.median(stacked, dim=0).values
        elif self.method == 'mean':
            combined = torch.mean(stacked, dim=0)
        elif self.method == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            combined = torch.einsum('m,mbt->bt', weights, stacked)
        elif self.method == 'learned':
            # (batch, n_steps, n_models)
            stacked_t = stacked.permute(1, 2, 0)
            combined = self.combiner(stacked_t).squeeze(-1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.bound_delta(combined)
    
    def freeze_base_models(self):
        """Freeze base models for ensemble weight training."""
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False


class VolatilityRegimeClassifier(nn.Module):
    """Classify volatility regime from features."""
    
    def __init__(self, input_dim: int, n_regimes: int = 3, hidden_size: int = 32):
        super().__init__()
        self.n_regimes = n_regimes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_regimes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return regime probabilities."""
        logits = self.classifier(features)
        return F.softmax(logits, dim=-1)


class RegimeAwareHedger(BaseHedgingModel):
    """
    Regime-aware hedger with volatility-based policy switching.
    
    Maintains separate sub-policies for different volatility regimes:
    - Low vol: conservative hedging
    - Normal vol: standard hedging
    - High vol: aggressive hedging
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        n_regimes: int = 3,
        delta_max: float = 1.5,
        dropout: float = 0.1
    ):
        super().__init__(delta_max)
        self.n_regimes = n_regimes
        
        # Regime classifier
        self.regime_classifier = VolatilityRegimeClassifier(
            input_dim=input_dim,
            n_regimes=n_regimes,
            hidden_size=32
        )
        
        # Sub-policies for each regime
        self.sub_policies = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
            for _ in range(n_regimes)
        ])
        
        # Output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(n_regimes)
        ])
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with regime-aware switching."""
        batch, n_steps, _ = features.shape
        device = features.device
        
        # Get regime probabilities for each timestep
        regime_probs = []
        for t in range(n_steps):
            probs = self.regime_classifier(features[:, t, :])
            regime_probs.append(probs)
        regime_probs = torch.stack(regime_probs, dim=1)  # (batch, n_steps, n_regimes)
        
        # Get outputs from each sub-policy
        sub_outputs = []
        for i, (lstm, head) in enumerate(zip(self.sub_policies, self.output_heads)):
            out, _ = lstm(features)
            delta = head(out).squeeze(-1)
            sub_outputs.append(delta)
        
        # Stack: (batch, n_steps, n_regimes)
        stacked = torch.stack(sub_outputs, dim=-1)
        
        # Weighted combination by regime probabilities
        combined = torch.sum(stacked * regime_probs, dim=-1)
        
        return self.bound_delta(combined)
    
    def get_regime_assignments(self, features: torch.Tensor) -> torch.Tensor:
        """Get hard regime assignments for analysis."""
        batch, n_steps, _ = features.shape
        
        regimes = []
        for t in range(n_steps):
            probs = self.regime_classifier(features[:, t, :])
            regime = torch.argmax(probs, dim=-1)
            regimes.append(regime)
        
        return torch.stack(regimes, dim=1)


class DistributionallyRobustHedger(BaseHedgingModel):
    """
    Distributionally robust hedger trained across multiple models.
    
    Trains on mixture of:
    - Heston (baseline)
    - Jump-diffusion
    - Rough volatility
    
    Uses worst-case optimization for robustness.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        delta_max: float = 1.5,
        dropout: float = 0.1
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        
        # Main hedging network
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Uncertainty-aware output
        self.mean_head = nn.Linear(hidden_size, 1)
        self.var_head = nn.Linear(hidden_size, 1)  # Epistemic uncertainty
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with uncertainty."""
        lstm_out, _ = self.lstm(features)
        mean = self.mean_head(lstm_out).squeeze(-1)
        
        return self.bound_delta(mean)
    
    def forward_with_uncertainty(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with epistemic uncertainty."""
        lstm_out, _ = self.lstm(features)
        mean = self.mean_head(lstm_out).squeeze(-1)
        log_var = self.var_head(lstm_out).squeeze(-1)
        var = F.softplus(log_var)
        
        return self.bound_delta(mean), var


class AdaptiveHedger(BaseHedgingModel):
    """
    Adaptive hedger that adjusts strategy based on market conditions.
    
    Uses attention to weight recent vs historical information
    based on detected market regime changes.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        adaptation_window: int = 5,
        delta_max: float = 1.5,
        dropout: float = 0.1
    ):
        super().__init__(delta_max)
        
        self.adaptation_window = adaptation_window
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Change detector
        self.change_detector = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptation gate
        self.adaptation_gate = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Sigmoid()
        )
        
        # Output
        self.output = nn.Linear(hidden_size, 1)
        
        self.apply(self.init_weights)
    
    def _detect_regime_change(self, features: torch.Tensor, t: int) -> torch.Tensor:
        """Detect if regime has changed."""
        if t < self.adaptation_window:
            return torch.zeros(features.size(0), 1, device=features.device)
        
        recent = features[:, t-self.adaptation_window:t, :].mean(dim=1)
        current = features[:, t, :]
        combined = torch.cat([recent, current], dim=-1)
        
        return self.change_detector(combined)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate adaptive deltas."""
        batch, n_steps, _ = features.shape
        
        lstm_out, _ = self.lstm(features)
        
        deltas = []
        for t in range(n_steps):
            h = lstm_out[:, t, :]
            
            # Detect regime change
            change_prob = self._detect_regime_change(features, t)
            
            # Adapt hidden state based on change detection
            gate_input = torch.cat([h, change_prob], dim=-1)
            gate = self.adaptation_gate(gate_input)
            adapted_h = gate * h
            
            delta = self.output(adapted_h)
            deltas.append(delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)


class HybridSignatureTransformer(BaseHedgingModel):
    """
    Combines signature features with transformer architecture.
    
    Uses signatures for pathwise information and transformer
    for temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        sig_order: int = 2,
        sig_window: int = 5,
        delta_max: float = 1.5,
        dropout: float = 0.1
    ):
        super().__init__(delta_max)
        
        from .signature_models import SignatureComputer
        
        # Signature computer
        self.sig_computer = SignatureComputer(
            input_dim=input_dim,
            sig_order=sig_order,
            window_size=sig_window,
            normalize=True
        )
        
        # Combined input projection
        combined_dim = input_dim + self.sig_computer.total_dim
        self.input_proj = nn.Linear(combined_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.output = nn.Linear(d_model, 1)
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with signature-transformer hybrid."""
        batch, n_steps, _ = features.shape
        
        # Compute signatures
        sig_features = []
        for t in range(n_steps):
            sig = self.sig_computer(features, t)
            sig_features.append(sig)
        sig_features = torch.stack(sig_features, dim=1)
        
        # Combine
        combined = torch.cat([features, sig_features], dim=-1)
        x = self.input_proj(combined)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :n_steps, :]
        
        # Causal mask
        mask = torch.triu(torch.ones(n_steps, n_steps, device=features.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Transformer
        encoded = self.transformer(x, mask=mask)
        
        # Output
        raw_deltas = self.output(encoded).squeeze(-1)
        
        return self.bound_delta(raw_deltas)
