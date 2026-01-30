"""
Candidate 5: Regime-Switching Ensemble (RSE)
=============================================

Grounded in: Gap analysis from existing pipeline

Key insight from AUDIT_SUMMARY.md:
- Transformer: Best for tail risk in volatile markets
- LSTM: Best in trending/stable markets  
- Signature models: Best for path-dependent exotics

Key idea: Dynamically weight model contributions based on detected market regime.

Architecture:
1. Regime Classifier: Detect current market regime from features
2. Base Models: Pre-trained LSTM, Transformer, SignatureMLP (frozen)
3. Gating Network: Map regime to model weights
4. Ensemble: Weighted combination of base model deltas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    n_regimes: int = 4  # Low vol, High vol, Trending, Mean-reverting
    window_size: int = 20
    vol_threshold: float = 0.02  # Annualized vol threshold
    trend_threshold: float = 0.01  # Trend threshold


class RegimeFeatureExtractor(nn.Module):
    """
    Extract regime-relevant features from market data.
    
    Features:
    - Realized volatility (different windows)
    - Trend strength (SMA crossover)
    - Jump indicator (Barndorff-Nielsen)
    - Mean reversion indicator
    """
    
    def __init__(self, windows: List[int] = [5, 10, 20]):
        super().__init__()
        self.windows = windows
    
    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Extract regime features.
        
        Args:
            prices: Price paths [B, T]
            
        Returns:
            features: Regime features [B, T, n_features]
        """
        B, T = prices.shape
        device = prices.device
        
        # Log returns
        log_prices = torch.log(prices + 1e-8)
        returns = log_prices[:, 1:] - log_prices[:, :-1]
        returns = F.pad(returns, (1, 0), value=0)  # [B, T]
        
        features = []
        
        # Realized volatility at different windows
        for w in self.windows:
            vol = self._rolling_std(returns, w)
            features.append(vol)
        
        # Trend indicator (SMA crossover)
        sma_short = self._rolling_mean(prices, 5)
        sma_long = self._rolling_mean(prices, 20)
        trend = (sma_short - sma_long) / (sma_long + 1e-8)
        features.append(trend)
        
        # Momentum
        momentum = (prices - self._shift(prices, 5)) / (self._shift(prices, 5) + 1e-8)
        features.append(momentum)
        
        # Jump indicator (simplified Barndorff-Nielsen)
        bv = self._rolling_bipower_variation(returns, 20)
        rv = self._rolling_std(returns, 20) ** 2
        jump_ratio = rv / (bv + 1e-8)
        features.append(jump_ratio)
        
        return torch.stack(features, dim=-1)  # [B, T, n_features]
    
    def _rolling_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Compute rolling mean."""
        B, T = x.shape
        result = torch.zeros_like(x)
        for t in range(T):
            start = max(0, t - window + 1)
            result[:, t] = x[:, start:t+1].mean(dim=1)
        return result
    
    def _rolling_std(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """Compute rolling standard deviation with NaN protection."""
        B, T = x.shape
        result = torch.zeros_like(x)
        for t in range(T):
            start = max(0, t - window + 1)
            if t - start >= 1:  # Need at least 2 samples for std
                std_val = x[:, start:t+1].std(dim=1)
                # Replace NaN with small value
                result[:, t] = torch.where(torch.isnan(std_val), torch.tensor(1e-6, device=x.device), std_val)
            else:
                result[:, t] = 1e-6
        return result
    
    def _rolling_bipower_variation(self, returns: torch.Tensor, window: int) -> torch.Tensor:
        """Compute rolling bipower variation for jump detection."""
        B, T = returns.shape
        result = torch.zeros_like(returns)
        for t in range(window, T):
            r = returns[:, t-window:t]
            bv = (r[:, :-1].abs() * r[:, 1:].abs()).sum(dim=1) * np.pi / 2
            result[:, t] = bv
        return result
    
    def _shift(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Shift tensor by n positions."""
        result = torch.zeros_like(x)
        if n > 0:
            result[:, n:] = x[:, :-n]
            result[:, :n] = x[:, 0:1]
        return result


class RegimeClassifier(nn.Module):
    """
    Neural network to classify market regime from features.
    
    Outputs probability distribution over K regimes.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        n_regimes: int = 4
    ):
        super().__init__()
        self.n_regimes = n_regimes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify regime from features.
        
        Args:
            features: Regime features [B, n_features] or [B, T, n_features]
            
        Returns:
            regime_probs: Regime probabilities [B, n_regimes] or [B, T, n_regimes]
        """
        if features.dim() == 3:
            B, T, d = features.shape
            features_flat = features.view(B * T, d)
            logits = self.net(features_flat)
            probs = F.softmax(logits, dim=-1)
            return probs.view(B, T, -1)
        else:
            logits = self.net(features)
            return F.softmax(logits, dim=-1)


class BaseModelWrapper(nn.Module):
    """Wrapper for base hedging models with frozen weights."""
    
    def __init__(self, model: nn.Module, freeze: bool = True):
        super().__init__()
        self.model = model
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SimpleLSTMHedger(nn.Module):
    """Simple LSTM hedger for ensemble."""
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 50,
        n_layers: int = 2,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.delta_max = delta_max
        
        self.lstm = nn.LSTM(
            input_dim + 1,  # +1 for previous delta
            hidden_dim,
            n_layers,
            batch_first=True
        )
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        device = x.device
        
        deltas = []
        h = None
        prev_delta = torch.zeros(B, 1, device=device)
        
        for t in range(T):
            inp = torch.cat([x[:, t:t+1, :], prev_delta.unsqueeze(1)], dim=-1)
            out, h = self.lstm(inp, h)
            delta = self.delta_max * torch.tanh(self.output(out[:, 0]))
            deltas.append(delta)
            prev_delta = delta
        
        return torch.cat(deltas, dim=-1)


class SimpleTransformerHedger(nn.Module):
    """Simple Transformer hedger for ensemble."""
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.delta_max = delta_max
        
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        h = self.input_proj(x)
        h = self.transformer(h, mask=mask)
        deltas = self.delta_max * torch.tanh(self.output(h))
        
        return deltas.squeeze(-1)


class SimpleSignatureHedger(nn.Module):
    """Simple signature-based hedger for ensemble."""
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.delta_max = delta_max
        
        # Simplified signature: use running statistics instead
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # current, mean, std
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        device = x.device
        
        deltas = []
        for t in range(T):
            # Running statistics as signature proxy
            path = x[:, :t+1, :]
            current = x[:, t, :]
            mean = path.mean(dim=1)
            if t >= 1:
                std = path.std(dim=1)
                std = torch.where(torch.isnan(std), torch.tensor(1e-6, device=device), std) + 1e-8
            else:
                std = torch.ones(B, d, device=device) * 1e-6
            
            features = torch.cat([current, mean, std], dim=-1)
            delta = self.delta_max * torch.tanh(self.feature_net(features))
            deltas.append(delta)
        
        return torch.cat(deltas, dim=-1)


class RegimeSwitchingEnsemble(nn.Module):
    """
    Regime-Switching Ensemble Hedger.
    
    Combines multiple base hedgers with regime-dependent weights.
    
    delta_t = sum_m w_m(regime_t) * delta_t^(m)
    
    Args:
        input_dim: Input feature dimension
        n_regimes: Number of market regimes
        delta_max: Maximum delta bound
        pretrained_models: Optional dict of pre-trained models
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        n_regimes: int = 4,
        delta_max: float = 1.5,
        pretrained_models: Optional[Dict[str, nn.Module]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_regimes = n_regimes
        self.delta_max = delta_max
        
        # Regime feature extractor
        self.feature_extractor = RegimeFeatureExtractor()
        
        # Regime classifier
        self.regime_classifier = RegimeClassifier(
            input_dim=6,  # Number of regime features
            n_regimes=n_regimes
        )
        
        # Base models (use pretrained if provided, else create new)
        if pretrained_models is not None:
            self.models = nn.ModuleDict({
                name: BaseModelWrapper(model, freeze=True)
                for name, model in pretrained_models.items()
            })
        else:
            self.models = nn.ModuleDict({
                'lstm': SimpleLSTMHedger(input_dim, delta_max=delta_max),
                'transformer': SimpleTransformerHedger(input_dim, delta_max=delta_max),
                'signature': SimpleSignatureHedger(input_dim, delta_max=delta_max)
            })
        
        n_models = len(self.models)
        
        # Regime-to-model affinity matrix [n_regimes, n_models]
        # Learnable mapping from regimes to model weights
        self.regime_model_affinity = nn.Parameter(
            torch.randn(n_regimes, n_models) * 0.1
        )
    
    def forward(
        self,
        features: torch.Tensor,
        prices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with regime-dependent ensemble.
        
        Args:
            features: Input features [B, T, d]
            prices: Price paths [B, T+1] for regime detection
            
        Returns:
            deltas: Ensemble hedging positions [B, T]
        """
        B, T, d = features.shape
        device = features.device
        
        # Extract regime features
        if prices is not None:
            regime_features = self.feature_extractor(prices[:, :-1])
        else:
            # Use first feature (log-moneyness) as price proxy
            regime_features = self.feature_extractor(features[:, :, 0])
        
        # Classify regime at each timestep
        regime_probs = self.regime_classifier(regime_features)  # [B, T, n_regimes]
        
        # Compute model weights from regime probabilities
        # w_m(t) = sum_r p(regime=r|t) * W_{r,m}
        affinity_probs = F.softmax(self.regime_model_affinity, dim=-1)  # [n_regimes, n_models]
        model_weights = torch.einsum('btr,rm->btm', regime_probs, affinity_probs)  # [B, T, n_models]
        
        # Get deltas from each base model
        model_deltas = []
        for name, model in self.models.items():
            delta = model(features)  # [B, T]
            model_deltas.append(delta)
        
        model_deltas = torch.stack(model_deltas, dim=-1)  # [B, T, n_models]
        
        # Weighted ensemble
        ensemble_deltas = (model_weights * model_deltas).sum(dim=-1)  # [B, T]
        
        # Clip to delta bounds
        ensemble_deltas = torch.clamp(ensemble_deltas, -self.delta_max, self.delta_max)
        
        return ensemble_deltas
    
    def get_regime_analysis(
        self,
        features: torch.Tensor,
        prices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Get detailed regime analysis for interpretability."""
        B, T, d = features.shape
        
        if prices is not None:
            regime_features = self.feature_extractor(prices[:, :-1])
        else:
            regime_features = self.feature_extractor(features[:, :, 0])
        
        regime_probs = self.regime_classifier(regime_features)
        
        affinity_probs = F.softmax(self.regime_model_affinity, dim=-1)
        model_weights = torch.einsum('btr,rm->btm', regime_probs, affinity_probs)
        
        return {
            'regime_probs': regime_probs,
            'model_weights': model_weights,
            'affinity_matrix': affinity_probs
        }


class RSETrainer:
    """
    Trainer for Regime-Switching Ensemble.
    
    Training protocol:
    1. Pre-train individual models on full dataset
    2. Freeze base models
    3. Train regime classifier and affinity matrix end-to-end
    """
    
    def __init__(
        self,
        model: RegimeSwitchingEnsemble,
        lr: float = 5e-4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        
        # Only train regime components (base models frozen)
        trainable_params = list(self.model.regime_classifier.parameters()) + \
                          [self.model.regime_model_affinity]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    def pretrain_base_models(
        self,
        train_loader,
        epochs: int = 30
    ):
        """Pre-train individual base models."""
        print("Pre-training base models...")
        
        for name, wrapper in self.model.models.items():
            print(f"  Training {name}...")
            
            # Unfreeze for pretraining
            for param in wrapper.model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=1e-3)
            
            for epoch in range(epochs):
                for batch in train_loader:
                    features = batch['features'].to(self.device)
                    prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
                    payoff = batch['payoff'].to(self.device)
                    
                    optimizer.zero_grad()
                    deltas = wrapper(features)
                    pnl = self._compute_pnl(deltas, prices, payoff)
                    loss = torch.log(torch.exp(-pnl).mean())
                    loss.backward()
                    optimizer.step()
            
            # Freeze after pretraining
            for param in wrapper.model.parameters():
                param.requires_grad = False
            
            print(f"    {name} pretrained (loss: {loss.item():.4f})")
    
    def train_gating(
        self,
        train_loader,
        epochs: int = 50
    ) -> Dict[str, list]:
        """Train regime classifier and affinity matrix."""
        print("Training gating network...")
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
                payoff = batch['payoff'].to(self.device)
                
                self.optimizer.zero_grad()
                deltas = self.model(features, prices)
                pnl = self._compute_pnl(deltas, prices, payoff)
                loss = torch.log(torch.exp(-pnl).mean())
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        return history
    
    def train_full(self, train_loader) -> Dict[str, Any]:
        """Full training: pretrain then train gating."""
        self.pretrain_base_models(train_loader)
        history = self.train_gating(train_loader)
        return history
    
    def train(self, train_loader, val_loader=None, epochs=80):
        """2-stage training interface for fair comparison with LSTM/Transformer.
        
        Stage 1: Pretrain base models (30 epochs each)
        Stage 2: Train gating network (remaining epochs)
        """
        # Stage 1: Pretrain base models (30 epochs for fairness)
        pretrain_epochs = min(30, epochs // 2)
        self.pretrain_base_models(train_loader, epochs=pretrain_epochs)
        
        # Stage 2: Train gating (remaining epochs)
        gating_epochs = epochs - pretrain_epochs
        if gating_epochs > 0:
            self.train_gating(train_loader, epochs=gating_epochs)
    
    def _compute_pnl(self, deltas, prices, payoff, tc=0.001):
        """Compute P&L."""
        price_changes = prices[:, 1:] - prices[:, :-1]
        hedge_gains = (deltas * price_changes).sum(dim=1)
        
        delta_changes = torch.zeros_like(deltas)
        delta_changes[:, 0] = deltas[:, 0]
        delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
        transaction_costs = (torch.abs(delta_changes) * prices[:, :-1] * tc).sum(dim=1)
        
        return -payoff + hedge_gains - transaction_costs


if __name__ == "__main__":
    print("Testing Regime-Switching Ensemble implementation...")
    
    # Create model
    model = RegimeSwitchingEnsemble(
        input_dim=5,
        n_regimes=4,
        delta_max=1.5
    )
    
    # Create dummy data
    B, T = 32, 30
    features = torch.randn(B, T, 5)
    prices = 100 + torch.cumsum(torch.randn(B, T + 1) * 0.5, dim=1)
    
    # Test forward pass
    deltas = model(features, prices)
    print(f"Deltas shape: {deltas.shape}")
    print(f"Delta range: [{deltas.min():.3f}, {deltas.max():.3f}]")
    
    # Test regime analysis
    analysis = model.get_regime_analysis(features, prices)
    print(f"Regime probs shape: {analysis['regime_probs'].shape}")
    print(f"Model weights shape: {analysis['model_weights'].shape}")
    print(f"Affinity matrix:\n{analysis['affinity_matrix']}")
    
    print("\n✅ Regime-Switching Ensemble implementation test passed!")
