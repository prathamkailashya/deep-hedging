"""
Hybrid Signature-Attention Hedger (H-SAH)

Novel architecture combining:
1. Time-augmented path signatures on rolling windows
2. Selective attention over signature levels (Level-1, Level-2, Level-3+)
3. Delta increment parametrization for smooth, bounded controls

Key innovations:
- Signature tokens as attention inputs (SigFormer-inspired)
- Each attention head specializes in different signature geometry
- Delta increments instead of direct delta output → admissible controls

References:
- Kidger et al. "Neural Rough Differential Equations for Long Time Series"
- Morrill et al. "A Generalised Signature Method for Multivariate Time Series"
- Kozyra "Deep Hedging with Recurrent Networks"
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class PathSignatureComputer(nn.Module):
    """
    Compute truncated path signatures on rolling windows.
    
    Time-augmented path: X_k = (t, S_t, log S_t, realized_vol, ΔS_t)
    Signature levels split for selective attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        window_size: int = 5,
        max_order: int = 3,
        augment_time: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.max_order = max_order
        self.augment_time = augment_time
        
        # Compute signature dimensions for each level
        d = input_dim + 1 if augment_time else input_dim
        self.level_dims = [d]  # Level 1
        for k in range(2, max_order + 1):
            self.level_dims.append(d ** k)
        
        self.total_sig_dim = sum(self.level_dims)
        self.d = d
    
    def _time_augment(self, path: torch.Tensor) -> torch.Tensor:
        """Add time coordinate as first channel."""
        batch, length, channels = path.shape
        time = torch.linspace(0, 1, length, device=path.device)
        time = time.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
        return torch.cat([time, path], dim=-1)
    
    def _compute_level1(self, path: torch.Tensor) -> torch.Tensor:
        """Level-1 signature: path increments (linear terms)."""
        return path[:, -1, :] - path[:, 0, :]
    
    def _compute_level2(self, path: torch.Tensor) -> torch.Tensor:
        """Level-2 signature: iterated integrals (area/interaction terms)."""
        batch, length, d = path.shape
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        # S^{ij} = Σ_k dX^i_k * X^j_{k-1}
        sig2 = []
        for i in range(d):
            for j in range(d):
                integral = torch.sum(increments[:, :, i] * path[:, :-1, j], dim=1)
                sig2.append(integral)
        
        return torch.stack(sig2, dim=1)
    
    def _compute_level3(self, path: torch.Tensor) -> torch.Tensor:
        """Level-3 signature: higher-order terms (roughness/memory)."""
        batch, length, d = path.shape
        increments = path[:, 1:, :] - path[:, :-1, :]
        
        # Approximate level-3 terms
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
    
    def compute_signature_levels(self, path: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute signature split by levels.
        
        Returns:
            [level1_sig, level2_sig, level3_sig, ...]
        """
        if self.augment_time:
            path = self._time_augment(path)
        
        levels = [self._compute_level1(path)]
        
        if self.max_order >= 2:
            levels.append(self._compute_level2(path))
        
        if self.max_order >= 3:
            levels.append(self._compute_level3(path))
        
        return levels
    
    def forward(self, features: torch.Tensor, t: int) -> List[torch.Tensor]:
        """
        Compute windowed signature at time t.
        
        Args:
            features: (batch, seq_len, input_dim)
            t: current timestep
            
        Returns:
            List of signature tensors for each level
        """
        batch = features.size(0)
        
        # Get window ending at t
        start = max(0, t - self.window_size + 1)
        window = features[:, start:t+1, :]
        
        # Pad if needed
        if window.size(1) < self.window_size:
            pad_size = self.window_size - window.size(1)
            padding = window[:, 0:1, :].expand(-1, pad_size, -1)
            window = torch.cat([padding, window], dim=1)
        
        return self.compute_signature_levels(window)


class SignatureLevelEmbedding(nn.Module):
    """Embed each signature level to common dimension."""
    
    def __init__(self, level_dims: List[int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            for dim in level_dims
        ])
    
    def forward(self, sig_levels: List[torch.Tensor]) -> torch.Tensor:
        """
        Embed signature levels and stack as tokens.
        
        Returns:
            tokens: (batch, n_levels, embed_dim)
        """
        embedded = [emb(sig) for emb, sig in zip(self.embeddings, sig_levels)]
        return torch.stack(embedded, dim=1)


class SelectiveSignatureAttention(nn.Module):
    """
    Attention over signature level tokens.
    
    Each attention head specializes in different signature geometry:
    - Head 1: Linear increments (trend)
    - Head 2: Area terms (volatility/correlation)
    - Head 3+: Roughness (path irregularity)
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        sig_tokens: torch.Tensor,
        state_query: torch.Tensor
    ) -> torch.Tensor:
        """
        Attend over signature tokens conditioned on current state.
        
        Args:
            sig_tokens: (batch, n_levels, embed_dim)
            state_query: (batch, embed_dim) - current market state
            
        Returns:
            attended: (batch, embed_dim)
        """
        batch, n_tokens, _ = sig_tokens.shape
        
        # Query from state, key-value from signatures
        Q = self.q_proj(state_query).unsqueeze(1)  # (batch, 1, embed_dim)
        K = self.k_proj(sig_tokens)  # (batch, n_tokens, embed_dim)
        V = self.v_proj(sig_tokens)
        
        # Multi-head reshape
        Q = Q.view(batch, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.matmul(attn, V)  # (batch, n_heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, self.embed_dim)
        
        # Output projection with residual
        out = self.out_proj(out)
        out = self.layer_norm(out + state_query)
        
        return out


class DeltaIncrementHead(nn.Module):
    """
    Output delta increments instead of direct deltas.
    
    Δδ_k = g_θ(attended_signature, state_k)
    δ_k = clip(δ_{k-1} + Δδ_k, [-δ_max, δ_max])
    
    This enforces:
    - Bounded controls by construction
    - Smooth trading (delta continuity)
    - Admissible hedging strategies
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        delta_max: float = 1.5,
        max_increment: float = 0.3
    ):
        super().__init__()
        self.delta_max = delta_max
        self.max_increment = max_increment
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for prev_delta
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        prev_delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute delta increment and new delta.
        
        Returns:
            delta: new bounded delta
            delta_increment: the change applied
        """
        x = torch.cat([features, prev_delta], dim=-1)
        raw_increment = self.network(x)
        
        # Bound increment
        delta_increment = self.max_increment * torch.tanh(raw_increment)
        
        # Apply increment and clip
        new_delta = prev_delta + delta_increment
        new_delta = torch.clamp(new_delta, -self.delta_max, self.delta_max)
        
        return new_delta.squeeze(-1), delta_increment.squeeze(-1)


class HybridSignatureAttentionHedger(nn.Module):
    """
    Hybrid Signature-Attention Hedger (H-SAH)
    
    Novel architecture for deep hedging with:
    1. Rolling window path signatures (time-augmented)
    2. Selective attention over signature levels
    3. Delta increment parametrization
    
    Key design principles:
    - Bounded controls (|δ| ≤ δ_max)
    - Smooth delta paths (no maturity spikes)
    - Trading volume O(1-3)
    - Risk-aware compatible
    """
    
    def __init__(
        self,
        input_dim: int,
        n_steps: int = 30,
        embed_dim: int = 64,
        n_heads: int = 3,
        sig_window: int = 5,
        sig_order: int = 3,
        delta_max: float = 1.5,
        max_increment: float = 0.3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        
        # Signature computation
        self.sig_computer = PathSignatureComputer(
            input_dim=input_dim,
            window_size=sig_window,
            max_order=sig_order,
            augment_time=True
        )
        
        # Signature level embeddings
        self.level_embed = SignatureLevelEmbedding(
            level_dims=self.sig_computer.level_dims,
            embed_dim=embed_dim
        )
        
        # State encoder (current market state → query)
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim + 1, embed_dim),  # +1 for prev_delta
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Selective attention over signatures
        self.sig_attention = SelectiveSignatureAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Delta increment head
        self.delta_head = DeltaIncrementHead(
            input_dim=embed_dim,
            hidden_dim=32,
            delta_max=delta_max,
            max_increment=max_increment
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate hedging deltas using signature attention.
        
        Args:
            features: (batch, n_steps, input_dim)
            
        Returns:
            deltas: (batch, n_steps)
        """
        batch, n_steps, _ = features.shape
        device = features.device
        
        deltas = []
        increments = []
        prev_delta = torch.zeros(batch, 1, device=device)
        
        for t in range(n_steps):
            # Compute signature levels for window ending at t
            sig_levels = self.sig_computer(features, t)
            
            # Embed signatures as tokens
            sig_tokens = self.level_embed(sig_levels)
            
            # Encode current state as query
            current_state = features[:, t, :]
            state_with_delta = torch.cat([current_state, prev_delta], dim=-1)
            state_query = self.state_encoder(state_with_delta)
            
            # Attend over signature tokens
            attended = self.sig_attention(sig_tokens, state_query)
            
            # Compute delta increment
            delta, increment = self.delta_head(attended, prev_delta)
            
            deltas.append(delta)
            increments.append(increment)
            prev_delta = delta.unsqueeze(-1)
        
        return torch.stack(deltas, dim=1)
    
    def forward_with_diagnostics(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with diagnostic information.
        
        Returns:
            deltas: (batch, n_steps)
            diagnostics: dict with increments, attention weights, etc.
        """
        batch, n_steps, _ = features.shape
        device = features.device
        
        deltas = []
        increments = []
        prev_delta = torch.zeros(batch, 1, device=device)
        
        for t in range(n_steps):
            sig_levels = self.sig_computer(features, t)
            sig_tokens = self.level_embed(sig_levels)
            
            current_state = features[:, t, :]
            state_with_delta = torch.cat([current_state, prev_delta], dim=-1)
            state_query = self.state_encoder(state_with_delta)
            
            attended = self.sig_attention(sig_tokens, state_query)
            delta, increment = self.delta_head(attended, prev_delta)
            
            deltas.append(delta)
            increments.append(increment)
            prev_delta = delta.unsqueeze(-1)
        
        deltas = torch.stack(deltas, dim=1)
        increments = torch.stack(increments, dim=1)
        
        diagnostics = {
            'increments': increments,
            'mean_abs_increment': increments.abs().mean(),
            'max_abs_delta': deltas.abs().max(),
            'delta_smoothness': (deltas[:, 1:] - deltas[:, :-1]).abs().mean()
        }
        
        return deltas, diagnostics


class HSAHTrainer:
    """
    Two-stage trainer for H-SAH model.
    
    Stage 1: CVaR pretraining (frictionless hedge)
    Stage 2: Entropic fine-tuning with trading penalties
    """
    
    def __init__(
        self,
        model: HybridSignatureAttentionHedger,
        lambda_risk: float = 1.0,
        gamma: float = 1e-3,
        band_width: float = 0.15,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.lambda_risk = lambda_risk
        self.gamma = gamma
        self.band_width = band_width
        self.device = device
        
        self.stage1_deltas = None  # Store for stage 2
    
    def _compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor,
        cost_multiplier: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute P&L and transaction costs."""
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        hedging_gains = torch.sum(deltas * price_changes, dim=1)
        
        # Transaction costs
        delta_changes = torch.cat([
            deltas[:, 0:1],
            deltas[:, 1:] - deltas[:, :-1],
            -deltas[:, -1:]
        ], dim=1)
        tc = cost_multiplier * torch.sum(torch.abs(delta_changes) * stock_paths, dim=1)
        
        pnl = -payoffs + hedging_gains - tc
        return pnl, tc
    
    def _cvar_loss(self, pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
        """CVaR loss."""
        losses = -pnl
        var = torch.quantile(losses, alpha)
        cvar = losses[losses >= var].mean()
        return cvar
    
    def _entropic_loss(self, pnl: torch.Tensor) -> torch.Tensor:
        """Entropic risk with numerical stability."""
        scaled = -self.lambda_risk * pnl
        max_val = scaled.max().detach()
        return (max_val + torch.log(torch.mean(torch.exp(scaled - max_val)))) / self.lambda_risk
    
    def _trading_penalty(self, deltas: torch.Tensor) -> torch.Tensor:
        """Penalty on delta changes."""
        changes = torch.abs(deltas[:, 1:] - deltas[:, :-1])
        return torch.mean(torch.sum(changes, dim=1))
    
    def _no_trade_band_penalty(
        self, 
        deltas: torch.Tensor,
        reference_deltas: torch.Tensor
    ) -> torch.Tensor:
        """Penalty for deviating outside no-trade band."""
        deviation = torch.abs(deltas - reference_deltas)
        outside_band = F.relu(deviation - self.band_width)
        return torch.mean(torch.sum(outside_band, dim=1))
    
    def train_stage1(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 30,
        lr: float = 0.001,
        patience: int = 10
    ) -> Dict:
        """Stage 1: CVaR pretraining."""
        print("\n" + "="*60)
        print("STAGE 1: CVaR Pretraining")
        print("="*60)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        from tqdm import tqdm
        pbar = tqdm(range(n_epochs), desc="Stage 1")
        
        for epoch in pbar:
            # Training
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
                loss = self._cvar_loss(pnl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = self.model(features)
                    pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
                    loss = self._cvar_loss(pnl)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        self.model.load_state_dict(best_state)
        
        # Store reference deltas for stage 2
        self.model.eval()
        all_deltas = []
        with torch.no_grad():
            for batch in train_loader:
                features = batch['features'].to(self.device)
                deltas = self.model(features)
                all_deltas.append(deltas.cpu())
        self.stage1_deltas = torch.cat(all_deltas, dim=0)
        
        return history
    
    def train_stage2(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 20,
        lr: float = 0.0001,
        patience: int = 10,
        nu: float = 1e8
    ) -> Dict:
        """Stage 2: Entropic fine-tuning with penalties."""
        print("\n" + "="*60)
        print("STAGE 2: Entropic Fine-tuning")
        print("="*60)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'entropic': [], 'penalty': []}
        
        from tqdm import tqdm
        pbar = tqdm(range(n_epochs), desc="Stage 2")
        
        batch_idx = 0
        for epoch in pbar:
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                # Get reference deltas
                batch_size = features.size(0)
                start = batch_idx * batch_size
                end = min(start + batch_size, len(self.stage1_deltas))
                ref_deltas = self.stage1_deltas[start:end].to(self.device)
                
                if ref_deltas.size(0) != batch_size:
                    batch_idx = 0
                    start = 0
                    end = batch_size
                    ref_deltas = self.stage1_deltas[start:end].to(self.device)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
                
                entropic = self._entropic_loss(pnl)
                trading_penalty = self._trading_penalty(deltas)
                band_penalty = self._no_trade_band_penalty(deltas, ref_deltas)
                
                loss = entropic + self.gamma * trading_penalty + nu * band_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                batch_idx += 1
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = self.model(features)
                    pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
                    loss = self._entropic_loss(pnl)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
            
            batch_idx = 0
        
        self.model.load_state_dict(best_state)
        return history
    
    def evaluate(self, test_loader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Evaluate model on test set."""
        self.model.eval()
        
        all_pnl = []
        all_deltas = []
        all_increments = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                deltas, diag = self.model.forward_with_diagnostics(features)
                pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
                
                all_pnl.append(pnl.cpu().numpy())
                all_deltas.append(deltas.cpu().numpy())
                all_increments.append(diag['increments'].cpu().numpy())
        
        pnl = np.concatenate(all_pnl)
        deltas = np.concatenate(all_deltas)
        increments = np.concatenate(all_increments)
        
        # Compute metrics
        losses = -pnl
        delta_changes = np.abs(np.diff(
            np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
            axis=1
        ))
        volume = np.mean(np.sum(delta_changes, axis=1))
        
        scaled = -self.lambda_risk * pnl
        max_val = np.max(scaled)
        entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / self.lambda_risk
        
        metrics = {
            'mean_pnl': np.mean(pnl),
            'std_pnl': np.std(pnl),
            'var_95': np.percentile(losses, 95),
            'var_99': np.percentile(losses, 99),
            'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
            'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)]),
            'entropic_risk': entropic,
            'trading_volume': volume,
            'max_delta': np.max(np.abs(deltas)),
            'mean_abs_increment': np.mean(np.abs(increments)),
            'delta_smoothness': np.mean(np.abs(np.diff(deltas, axis=1)))
        }
        
        return metrics, pnl, deltas
