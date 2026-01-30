"""
Simplified Hybrid Model: Signature-Augmented LSTM with Delta Increments

Key insight from ablation: Complex attention over signatures hurts performance.
This simplified approach:
1. Computes rolling signatures as ADDITIONAL features (not replacement)
2. Uses standard LSTM architecture (proven to work)
3. Outputs delta increments for smooth control
4. Two-stage training (CVaR → Entropic)

This is the "minimal viable hybrid" that should beat LSTM on tail risk.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class RollingSignatureFeatures(nn.Module):
    """
    Compute rolling signature features efficiently.
    
    Uses level-1 and level-2 signatures only (higher orders add noise).
    """
    
    def __init__(self, input_dim: int, window_size: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        
        # Output: level-1 (d) + level-2 (d^2) + time-augmented versions
        d = input_dim + 1  # +1 for time
        self.sig_dim = d + d * d  # Level 1 + Level 2
    
    def _compute_signature(self, window: torch.Tensor) -> torch.Tensor:
        """Compute signature of a single window."""
        batch, length, d = window.shape
        
        # Level 1: increments
        level1 = window[:, -1, :] - window[:, 0, :]
        
        # Level 2: iterated integrals (simplified)
        increments = window[:, 1:, :] - window[:, :-1, :]
        
        level2 = []
        for i in range(d):
            for j in range(d):
                integral = torch.sum(increments[:, :, i] * window[:, :-1, j], dim=1)
                level2.append(integral)
        level2 = torch.stack(level2, dim=1)
        
        return torch.cat([level1, level2], dim=1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute rolling signatures for all timesteps.
        
        Args:
            features: (batch, seq_len, input_dim)
            
        Returns:
            sig_features: (batch, seq_len, sig_dim)
        """
        batch, seq_len, input_dim = features.shape
        device = features.device
        
        # Add time coordinate
        time = torch.linspace(0, 1, seq_len, device=device)
        time = time.unsqueeze(0).unsqueeze(-1).expand(batch, seq_len, 1)
        augmented = torch.cat([time, features], dim=-1)
        
        # Pad for initial windows
        padding = augmented[:, 0:1, :].expand(-1, self.window_size - 1, -1)
        padded = torch.cat([padding, augmented], dim=1)
        
        # Compute signatures for each window
        signatures = []
        for t in range(seq_len):
            window = padded[:, t:t + self.window_size, :]
            sig = self._compute_signature(window)
            signatures.append(sig)
        
        return torch.stack(signatures, dim=1)


class HybridSigLSTM(nn.Module):
    """
    Hybrid Signature-LSTM with Delta Increments.
    
    Architecture:
    1. Compute rolling signature features
    2. Concatenate with original features
    3. Process with LSTM
    4. Output delta increments (bounded)
    
    Key design choices:
    - Delta increments instead of direct delta → smooth trading
    - Signature features → pathwise information
    - Standard LSTM → proven architecture
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        sig_window: int = 5,
        delta_max: float = 1.5,
        max_increment: float = 0.2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.delta_max = delta_max
        self.max_increment = max_increment
        self.hidden_size = hidden_size
        
        # Signature feature extractor
        self.sig_features = RollingSignatureFeatures(input_dim, sig_window)
        
        # Combined input dimension
        combined_dim = input_dim + self.sig_features.sig_dim + 1  # +1 for prev_delta
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Delta increment head
        self.increment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.increment_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate hedging deltas with increment-based control.
        
        Args:
            features: (batch, seq_len, input_dim)
            
        Returns:
            deltas: (batch, seq_len)
        """
        batch, seq_len, _ = features.shape
        device = features.device
        
        # Compute signature features
        sig_feats = self.sig_features(features)
        
        # Process sequentially with delta feedback
        deltas = []
        increments = []
        prev_delta = torch.zeros(batch, 1, device=device)
        hidden = None
        
        for t in range(seq_len):
            # Combine features
            combined = torch.cat([
                features[:, t:t+1, :],
                sig_feats[:, t:t+1, :],
                prev_delta.unsqueeze(1)
            ], dim=-1)
            
            # LSTM step
            output, hidden = self.lstm(combined, hidden)
            
            # Compute delta increment
            raw_increment = self.increment_head(output.squeeze(1))
            increment = self.max_increment * torch.tanh(raw_increment)
            
            # Apply increment with clipping
            new_delta = torch.clamp(prev_delta + increment, -self.delta_max, self.delta_max)
            
            deltas.append(new_delta.squeeze(-1))
            increments.append(increment.squeeze(-1))
            prev_delta = new_delta
        
        return torch.stack(deltas, dim=1)
    
    def forward_with_increments(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both deltas and increments."""
        batch, seq_len, _ = features.shape
        device = features.device
        
        sig_feats = self.sig_features(features)
        
        deltas = []
        increments = []
        prev_delta = torch.zeros(batch, 1, device=device)
        hidden = None
        
        for t in range(seq_len):
            combined = torch.cat([
                features[:, t:t+1, :],
                sig_feats[:, t:t+1, :],
                prev_delta.unsqueeze(1)
            ], dim=-1)
            
            output, hidden = self.lstm(combined, hidden)
            raw_increment = self.increment_head(output.squeeze(1))
            increment = self.max_increment * torch.tanh(raw_increment)
            new_delta = torch.clamp(prev_delta + increment, -self.delta_max, self.delta_max)
            
            deltas.append(new_delta.squeeze(-1))
            increments.append(increment.squeeze(-1))
            prev_delta = new_delta
        
        return torch.stack(deltas, dim=1), torch.stack(increments, dim=1)


class HybridTrainer:
    """
    Two-stage trainer for Hybrid model.
    
    Stage 1: CVaR loss (learn good frictionless hedge)
    Stage 2: Entropic loss + increment penalty (smooth trading)
    """
    
    def __init__(
        self,
        model: HybridSigLSTM,
        lambda_risk: float = 1.0,
        increment_penalty: float = 0.01,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.lambda_risk = lambda_risk
        self.increment_penalty = increment_penalty
        self.device = device
    
    def _compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor
    ) -> torch.Tensor:
        """Compute P&L."""
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        hedging_gains = torch.sum(deltas * price_changes, dim=1)
        return -payoffs + hedging_gains
    
    def _cvar_loss(self, pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
        """CVaR loss."""
        losses = -pnl
        var = torch.quantile(losses, alpha)
        cvar = losses[losses >= var].mean()
        return cvar
    
    def _entropic_loss(self, pnl: torch.Tensor) -> torch.Tensor:
        """Entropic risk (numerically stable)."""
        scaled = -self.lambda_risk * pnl
        max_val = scaled.max().detach()
        return (max_val + torch.log(torch.mean(torch.exp(scaled - max_val)))) / self.lambda_risk
    
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
        
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = []
        
        from tqdm import tqdm
        pbar = tqdm(range(n_epochs), desc="Stage 1")
        
        for epoch in pbar:
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                pnl = self._compute_pnl(deltas, stock_paths, payoffs)
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
                    pnl = self._compute_pnl(deltas, stock_paths, payoffs)
                    loss = self._cvar_loss(pnl)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history.append({'train': train_loss, 'val': val_loss})
            
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
        return history
    
    def train_stage2(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 20,
        lr: float = 0.0001,
        patience: int = 10
    ) -> Dict:
        """Stage 2: Entropic fine-tuning with increment penalty."""
        print("\n" + "="*60)
        print("STAGE 2: Entropic Fine-tuning")
        print("="*60)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = []
        
        from tqdm import tqdm
        pbar = tqdm(range(n_epochs), desc="Stage 2")
        
        for epoch in pbar:
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas, increments = self.model.forward_with_increments(features)
                pnl = self._compute_pnl(deltas, stock_paths, payoffs)
                
                entropic = self._entropic_loss(pnl)
                inc_penalty = self.increment_penalty * increments.abs().mean()
                
                loss = entropic + inc_penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation (entropic only)
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = self.model(features)
                    pnl = self._compute_pnl(deltas, stock_paths, payoffs)
                    loss = self._entropic_loss(pnl)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history.append({'train': train_loss, 'val': val_loss})
            
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
        return history
    
    def evaluate(self, test_loader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Evaluate on test set."""
        self.model.eval()
        
        all_pnl = []
        all_deltas = []
        all_increments = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                deltas, increments = self.model.forward_with_increments(features)
                pnl = self._compute_pnl(deltas, stock_paths, payoffs)
                
                all_pnl.append(pnl.cpu().numpy())
                all_deltas.append(deltas.cpu().numpy())
                all_increments.append(increments.cpu().numpy())
        
        pnl = np.concatenate(all_pnl)
        deltas = np.concatenate(all_deltas)
        increments = np.concatenate(all_increments)
        
        # Metrics
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
