"""
Candidate 2: Rough Volatility Signature Network (RVSN)
=======================================================

Grounded in: Abi Jaber & Gérard 2025 "Hedging with memory: shallow and deep learning with signatures"

Key idea: Signatures excel on non-Markovian (rough volatility) paths, not Heston.
The existing pipeline tested signatures on Heston (Markovian) - wrong market model.

This implementation:
1. Adds rough Bergomi (rBergomi) volatility simulator
2. Implements adaptive signature truncation based on local Hurst estimate
3. Uses weighted signatures for regime-adaptive hedging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class RoughBergomiParams:
    """Parameters for rough Bergomi model."""
    S0: float = 100.0       # Initial stock price
    xi: float = 0.04        # Initial forward variance
    eta: float = 1.9        # Vol-of-vol
    H: float = 0.1          # Hurst parameter (0 < H < 0.5 for rough)
    rho: float = -0.7       # Correlation
    r: float = 0.05         # Risk-free rate
    T: float = 30/365       # Time to maturity
    n_steps: int = 30       # Number of time steps


class RoughVolatilitySimulator:
    """
    Rough Bergomi model simulator for non-Markovian volatility.
    
    The rough Bergomi model is defined as:
        dS_t = S_t * sqrt(V_t) * dW_t
        V_t = xi * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})
    
    where W^H is fractional Brownian motion with Hurst parameter H.
    
    For H < 0.5, the model exhibits rough volatility behavior observed
    in real markets (H ≈ 0.1 empirically).
    """
    
    def __init__(self, params: RoughBergomiParams):
        self.params = params
    
    def simulate_fbm(
        self,
        n_paths: int,
        n_steps: int,
        H: float,
        T: float
    ) -> torch.Tensor:
        """
        Simulate fractional Brownian motion using Cholesky decomposition.
        
        Args:
            n_paths: Number of paths
            n_steps: Number of time steps
            H: Hurst parameter
            T: Total time
            
        Returns:
            fBm paths [n_paths, n_steps+1]
        """
        dt = T / n_steps
        times = torch.linspace(0, T, n_steps + 1)
        
        # Covariance matrix for fBm
        # Cov(W^H_s, W^H_t) = 0.5 * (|s|^{2H} + |t|^{2H} - |t-s|^{2H})
        cov = torch.zeros(n_steps + 1, n_steps + 1)
        for i in range(n_steps + 1):
            for j in range(n_steps + 1):
                s, t = times[i], times[j]
                if s > 0 or t > 0:
                    cov[i, j] = 0.5 * (
                        s.abs() ** (2*H) + 
                        t.abs() ** (2*H) - 
                        (t - s).abs() ** (2*H)
                    )
        
        # Cholesky decomposition
        cov = cov + 1e-6 * torch.eye(n_steps + 1)  # Regularization
        L = torch.linalg.cholesky(cov)
        
        # Generate standard normal and transform
        Z = torch.randn(n_paths, n_steps + 1)
        W_H = Z @ L.T
        
        return W_H
    
    def simulate_paths(
        self,
        n_paths: int,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate rough Bergomi paths.
        
        Returns:
            prices: Stock price paths [n_paths, n_steps+1]
            variance: Variance process [n_paths, n_steps+1]
            fbm: Fractional Brownian motion [n_paths, n_steps+1]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        p = self.params
        n_steps = p.n_steps
        dt = p.T / n_steps
        
        # Simulate fBm for variance
        W_H = self.simulate_fbm(n_paths, n_steps, p.H, p.T)
        
        # Simulate correlated Brownian motion for price
        Z1 = torch.randn(n_paths, n_steps)
        Z2 = torch.randn(n_paths, n_steps)
        dW_S = p.rho * (W_H[:, 1:] - W_H[:, :-1]) + np.sqrt(1 - p.rho**2) * Z2 * np.sqrt(dt)
        
        # Variance process
        times = torch.linspace(0, p.T, n_steps + 1)
        variance = p.xi * torch.exp(
            p.eta * W_H - 0.5 * p.eta**2 * times.unsqueeze(0) ** (2 * p.H)
        )
        
        # Price process (Euler-Maruyama)
        prices = torch.zeros(n_paths, n_steps + 1)
        prices[:, 0] = p.S0
        
        for t in range(n_steps):
            vol = torch.sqrt(torch.clamp(variance[:, t], min=1e-8))
            prices[:, t+1] = prices[:, t] * torch.exp(
                (p.r - 0.5 * variance[:, t]) * dt + vol * dW_S[:, t]
            )
        
        return prices, variance, W_H
    
    def generate_hedging_data(
        self,
        n_paths: int,
        strike: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate complete hedging dataset."""
        prices, variance, fbm = self.simulate_paths(n_paths, seed)
        
        p = self.params
        strike = strike or p.S0
        
        # Time to maturity at each step
        times = torch.linspace(0, p.T, p.n_steps + 1)
        ttm = p.T - times  # [n_steps+1]
        ttm = ttm.unsqueeze(0).expand(n_paths, -1)  # [n_paths, n_steps+1]
        
        # Option payoff (European call)
        payoff = F.relu(prices[:, -1] - strike)
        
        # Features: [log-moneyness, sqrt(variance), ttm, normalized_price, fbm]
        log_moneyness = torch.log(prices / strike)
        sqrt_var = torch.sqrt(torch.clamp(variance, min=1e-8))
        norm_price = prices / p.S0
        
        features = torch.stack([
            log_moneyness[:, :-1],
            sqrt_var[:, :-1],
            ttm[:, :-1],
            norm_price[:, :-1],
            fbm[:, :-1]
        ], dim=-1)  # [n_paths, n_steps, 5]
        
        return {
            'features': features,
            'prices': prices,
            'variance': variance,
            'payoff': payoff,
            'fbm': fbm
        }


class HurstEstimator(nn.Module):
    """
    Neural network to estimate local Hurst parameter from returns.
    
    Uses realized variation ratios to estimate roughness of the path.
    """
    
    def __init__(self, window_size: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.window_size = window_size
        self.mlp = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in (0, 1), scale to (0, 0.5) for rough regime
        )
    
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Estimate local Hurst parameter.
        
        Args:
            returns: Log returns [B, T]
            
        Returns:
            H_local: Local Hurst estimates [B, T] in (0, 0.5)
        """
        B, T = returns.shape
        H_local = torch.zeros(B, T, device=returns.device)
        
        for t in range(self.window_size, T):
            window = returns[:, t-self.window_size:t]
            H_local[:, t] = 0.5 * self.mlp(window).squeeze(-1)  # Scale to (0, 0.5)
        
        # Fill early timesteps with mean
        H_local[:, :self.window_size] = H_local[:, self.window_size:self.window_size+1]
        
        return H_local


class AdaptiveSignatureHedger(nn.Module):
    """
    Signature-based hedger with adaptive truncation order.
    
    Key innovation: Learn optimal signature truncation order per market regime
    based on local Hurst estimate and volatility.
    
    Higher-order signatures are needed for rougher paths (lower H).
    
    Args:
        input_dim: Input feature dimension
        max_depth: Maximum signature truncation order
        hidden_dim: Hidden layer dimension
        delta_max: Maximum delta bound
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        max_depth: int = 4,
        hidden_dim: int = 64,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_depth = max_depth
        self.delta_max = delta_max
        
        # Signature dimensions for each depth
        self.sig_dims = self._compute_sig_dims(input_dim, max_depth)
        
        # Hurst estimator
        self.hurst_estimator = HurstEstimator(window_size=10)
        
        # Gating network: [H_local, vol_local] -> weights over depths
        self.gating = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, max_depth)
        )
        
        # Signature projections to common dimension
        self.sig_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in self.sig_dims
        ])
        
        # Hedging MLP
        self.hedger = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _compute_sig_dims(self, d: int, max_depth: int) -> List[int]:
        """Compute signature dimension for each truncation depth."""
        dims = []
        for depth in range(1, max_depth + 1):
            # Signature of depth k has dimension sum_{i=1}^{k} d^i
            dim = sum(d ** i for i in range(1, depth + 1))
            dims.append(dim)
        return dims
    
    def _compute_signatures(
        self,
        path: torch.Tensor,
        depth: int
    ) -> torch.Tensor:
        """
        Compute path signature up to given depth.
        
        Simplified implementation - in production, use signatory library.
        
        Args:
            path: Input path [B, T, d]
            depth: Truncation depth
            
        Returns:
            signature: Path signature [B, sig_dim]
        """
        B, T, d = path.shape
        
        # Level 1: Increment
        increments = path[:, -1] - path[:, 0]  # [B, d]
        
        if depth == 1:
            return increments
        
        # Level 2: Iterated integrals (simplified)
        sig = [increments]
        
        if depth >= 2:
            # Approximate level 2
            level2 = torch.zeros(B, d * d, device=path.device)
            for i in range(T - 1):
                inc_i = path[:, i+1] - path[:, i]
                for j in range(i + 1, T):
                    inc_j = path[:, j] - path[:, j-1] if j > 0 else path[:, j]
                    level2 += torch.einsum('bi,bj->bij', inc_i, inc_j).view(B, -1)
            sig.append(level2)
        
        if depth >= 3:
            # Level 3: Triple iterated integrals
            level3 = torch.zeros(B, d * d * d, device=path.device)
            for i in range(T - 2):
                inc_i = path[:, i+1] - path[:, i]
                for j in range(i + 1, T - 1):
                    inc_j = path[:, j+1] - path[:, j]
                    for k in range(j + 1, T):
                        inc_k = path[:, k] - path[:, k-1] if k > 0 else path[:, k]
                        # Outer product of three increments
                        triple = torch.einsum('bi,bj,bk->bijk', inc_i, inc_j, inc_k)
                        level3 += triple.view(B, -1)
            sig.append(level3)
        
        if depth >= 4:
            # Level 4: Quadruple iterated integrals (computationally expensive)
            level4 = torch.zeros(B, d * d * d * d, device=path.device)
            # Use Chen's relation for efficiency: approximate via lower levels
            # S^4 ≈ S^2 ⊗ S^2 (shuffle product approximation)
            if len(sig) >= 2:
                level2 = sig[1]
                level4 = torch.einsum('bi,bj->bij', level2, level2).view(B, -1)[:, :d**4]
                # Pad if needed
                if level4.shape[1] < d**4:
                    level4 = F.pad(level4, (0, d**4 - level4.shape[1]))
            sig.append(level4)
        
        return torch.cat(sig, dim=-1)
    
    def forward(
        self,
        features: torch.Tensor,
        prev_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive signature computation.
        
        Args:
            features: Input features [B, T, d]
            prev_delta: Previous delta (for recurrent connection)
            
        Returns:
            deltas: Hedging positions [B, T]
        """
        B, T, d = features.shape
        device = features.device
        
        # Compute returns for Hurst estimation
        if d > 0:
            log_prices = features[:, :, 0]  # Assuming first feature is log-moneyness
            returns = log_prices[:, 1:] - log_prices[:, :-1]
            returns = F.pad(returns, (1, 0), value=0)
        else:
            returns = torch.zeros(B, T, device=device)
        
        # Estimate local Hurst
        H_local = self.hurst_estimator(returns)  # [B, T]
        
        # Compute local volatility
        vol_local = returns.abs().cumsum(dim=1) / (torch.arange(1, T+1, device=device).float() + 1e-8)
        
        # Compute signatures at each depth (using full path up to t)
        deltas = []
        
        for t in range(T):
            # Path up to time t (minimum 2 points for meaningful signature)
            start_idx = max(0, t - 10)  # Use rolling window of 10 steps
            path_t = features[:, start_idx:t+1, :]  # [B, window, d]
            
            # Gating weights based on local regime
            regime_features = torch.stack([H_local[:, t], vol_local[:, t]], dim=-1)
            gate_logits = self.gating(regime_features)  # [B, max_depth]
            gate_weights = F.softmax(gate_logits, dim=-1)  # [B, max_depth]
            
            # Compute weighted signature representation
            sig_repr = torch.zeros(B, self.sig_projections[0].out_features, device=device)
            
            path_len = path_t.shape[1]
            for depth in range(self.max_depth):
                if path_len >= depth + 1:  # Need enough points for signature
                    sig = self._compute_signatures(path_t, depth + 1)
                    # Ensure correct size
                    expected_dim = self.sig_dims[depth]
                    if sig.shape[-1] < expected_dim:
                        sig = F.pad(sig, (0, expected_dim - sig.shape[-1]))
                    elif sig.shape[-1] > expected_dim:
                        sig = sig[:, :expected_dim]
                    sig_proj = self.sig_projections[depth](sig)
                    sig_repr += gate_weights[:, depth:depth+1] * sig_proj
            
            # Combine with current features
            current_features = features[:, t, :]
            combined = torch.cat([sig_repr, current_features], dim=-1)
            
            # Predict delta
            delta_t = self.delta_max * torch.tanh(self.hedger(combined))
            deltas.append(delta_t)
        
        return torch.cat(deltas, dim=-1)  # [B, T]


class RSVNTrainer:
    """Trainer for Rough Volatility Signature Network.
    
    Uses proper 2-stage training for FAIR comparison with LSTM/Transformer:
    Stage 1: CVaR pretraining (tail risk awareness)
    Stage 2: Entropic fine-tuning (utility maximization)
    """
    
    def __init__(
        self,
        model: AdaptiveSignatureHedger,
        lr: float = 1e-3,  # Match LSTM/Transformer LR
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss functions for 2-stage training
        self.cvar_loss_fn = self._cvar_loss
        self.entropic_loss_fn = self._entropic_loss
    
    def _cvar_loss(self, pnl, alpha=0.95):
        """CVaR loss for Stage 1."""
        sorted_pnl, _ = torch.sort(pnl)
        k = int((1 - alpha) * len(pnl))
        k = max(1, k)  # At least 1 sample
        return -sorted_pnl[:k].mean()  # CVaR of negative P&L
    
    def _entropic_loss(self, pnl, lambda_risk=1.0):
        """Entropic loss for Stage 2."""
        return torch.log(torch.exp(-lambda_risk * pnl).mean()) / lambda_risk
    
    def train_epoch(self, train_loader, loss_fn) -> float:
        """Train for one epoch with specified loss function."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
            payoff = batch['payoff'].to(self.device)
            
            self.optimizer.zero_grad()
            deltas = self.model(features)
            pnl = self._compute_pnl(deltas, prices, payoff)
            
            loss = loss_fn(pnl)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def train(self, train_loader, val_loader=None, epochs=80):
        """2-stage training matching LSTM/Transformer protocol.
        
        Stage 1 (50 epochs): CVaR pretraining with LR=1e-3
        Stage 2 (30 epochs): Entropic fine-tuning with LR=1e-4
        
        This ensures FAIR comparison - same curriculum as baselines.
        """
        stage1_epochs = 50 if epochs >= 50 else epochs
        stage2_epochs = max(0, epochs - 50)
        
        # Stage 1: CVaR pretraining (CRITICAL for fair comparison)
        print(f"Stage 1: CVaR Pretraining ({stage1_epochs} epochs)")
        for epoch in range(stage1_epochs):
            loss = self.train_epoch(train_loader, self.cvar_loss_fn)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{stage1_epochs}: CVaR Loss = {loss:.4f}")
        
        # Stage 2: Entropic fine-tuning with reduced LR
        if stage2_epochs > 0:
            print(f"Stage 2: Entropic Fine-tuning ({stage2_epochs} epochs)")
            # Reduce learning rate (critical for fair comparison)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1  # 1e-3 -> 1e-4
            
            for epoch in range(stage2_epochs):
                loss = self.train_epoch(train_loader, self.entropic_loss_fn)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1}/{stage2_epochs}: Entropic Loss = {loss:.4f}")
    
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
    print("Testing RVSN implementation...")
    
    # Test rough volatility simulator
    params = RoughBergomiParams(H=0.1)  # Rough regime
    simulator = RoughVolatilitySimulator(params)
    
    data = simulator.generate_hedging_data(n_paths=100, seed=42)
    print(f"Features shape: {data['features'].shape}")
    print(f"Prices shape: {data['prices'].shape}")
    print(f"Variance range: [{data['variance'].min():.4f}, {data['variance'].max():.4f}]")
    
    # Test adaptive signature hedger
    model = AdaptiveSignatureHedger(
        input_dim=5,
        max_depth=4,
        hidden_dim=64
    )
    
    deltas = model(data['features'])
    print(f"Deltas shape: {deltas.shape}")
    print(f"Delta range: [{deltas.min():.3f}, {deltas.max():.3f}]")
    
    print("\n✅ RVSN implementation test passed!")
