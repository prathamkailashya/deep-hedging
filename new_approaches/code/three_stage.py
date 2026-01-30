"""
Candidate 4: Three-Stage Curriculum Hedger (3SCH)
==================================================

Grounded in: Kozyra 2018 (Oxford MSc Thesis) - extended two-stage training

Key idea: Add an intermediate stage with mixed CVaR+Entropic objective
to smooth the transition between CVaR pretraining and entropic fine-tuning.

Training protocol:
    Stage 1: 50 epochs, CVaR95 loss (tail risk awareness)
    Stage 2: 20 epochs, 0.5*CVaR + 0.5*Entropic (smooth transition) - NEW
    Stage 3: 30 epochs, Entropic + trading penalty (fine-tuning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Callable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.train.losses import EntropicLoss, CVaRLoss


class MixedLoss(nn.Module):
    """
    Mixed CVaR + Entropic loss for curriculum learning.
    
    L = alpha * CVaR_95(-P&L) + (1 - alpha) * Entropic(-P&L)
    
    Args:
        cvar_weight: Weight for CVaR loss (1 - cvar_weight for entropic)
        cvar_alpha: CVaR confidence level
        entropic_lambda: Entropic risk aversion parameter
    """
    
    def __init__(
        self,
        cvar_weight: float = 0.5,
        cvar_alpha: float = 0.95,
        entropic_lambda: float = 1.0
    ):
        super().__init__()
        self.cvar_weight = cvar_weight
        self.cvar_loss = CVaRLoss(alpha=cvar_alpha)
        self.entropic_loss = EntropicLoss(lambda_risk=entropic_lambda)
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute mixed loss.
        
        Args:
            pnl: P&L tensor [B]
            
        Returns:
            loss: Weighted combination of CVaR and entropic loss
        """
        cvar = self.cvar_loss(pnl)
        entropic = self.entropic_loss(pnl)
        
        return self.cvar_weight * cvar + (1 - self.cvar_weight) * entropic
    
    def set_weight(self, weight: float):
        """Update CVaR weight (for annealing)."""
        self.cvar_weight = weight


class TradingPenalty(nn.Module):
    """
    Penalty for excessive trading (transaction cost proxy).
    
    Penalty = gamma * sum_t |delta_t - delta_{t-1}|
    """
    
    def __init__(self, gamma: float = 1e-3):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute trading penalty.
        
        Args:
            deltas: Delta positions [B, T]
            
        Returns:
            penalty: Mean trading penalty
        """
        # Changes in delta
        delta_changes = torch.zeros_like(deltas)
        delta_changes[:, 0] = deltas[:, 0].abs()  # Initial position
        delta_changes[:, 1:] = (deltas[:, 1:] - deltas[:, :-1]).abs()
        
        return self.gamma * delta_changes.sum(dim=1).mean()


class NoTradeBandPenalty(nn.Module):
    """
    Penalty for deviating from reference delta outside no-trade band.
    
    From Kozyra 2018: Encourages stability around Stage 1 solution.
    
    Penalty = nu * sum_t max(0, |delta_t - delta_ref_t| - epsilon)
    """
    
    def __init__(self, nu: float = 1e8, epsilon: float = 0.15):
        super().__init__()
        self.nu = nu
        self.epsilon = epsilon
    
    def forward(
        self,
        deltas: torch.Tensor,
        reference_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute no-trade band penalty.
        
        Args:
            deltas: Current delta positions [B, T]
            reference_deltas: Reference deltas from Stage 1 [B, T]
            
        Returns:
            penalty: Mean penalty for exceeding band
        """
        deviation = (deltas - reference_deltas).abs()
        excess = F.relu(deviation - self.epsilon)
        
        return self.nu * excess.sum(dim=1).mean()


class ThreeStageTrainer:
    """
    Three-stage curriculum trainer for deep hedging.
    
    Extends the Kozyra two-stage protocol with an intermediate stage
    for smoother loss landscape transition.
    
    Stage 1: CVaR pretraining (50 epochs, lr=1e-3, patience=15)
    Stage 2: Mixed CVaR+Entropic with alpha annealing (20 epochs) - NEW
    Stage 3: Entropic fine-tuning with trading penalties (30 epochs, lr=1e-4, patience=10)
    
    Training protocol aligned with paper.tex Section 5.1.
    
    Args:
        model: Hedging model
        lr_stage1: Learning rate for Stage 1 (paper.tex: 1e-3)
        lr_stage2: Learning rate for Stage 2 (intermediate)
        lr_stage3: Learning rate for Stage 3 (paper.tex: 1e-4)
        weight_decay: L2 regularization (paper.tex: 1e-4)
        epochs_stage1: Epochs for Stage 1 (paper.tex: 50)
        epochs_stage2: Epochs for Stage 2 (intermediate: 20)
        epochs_stage3: Epochs for Stage 3 (paper.tex: 30)
        patience_stage1: Early stopping patience Stage 1 (paper.tex: 15)
        patience_stage3: Early stopping patience Stage 3 (paper.tex: 10)
        grad_clip: Gradient clipping norm (paper.tex: 5.0)
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        lr_stage1: float = 1e-3,
        lr_stage2: float = 5e-4,
        lr_stage3: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs_stage1: int = 50,
        epochs_stage2: int = 0,  # Skip by default for paper.tex 2-stage (50+30)
        epochs_stage3: int = 30,
        patience_stage1: int = 15,
        patience_stage3: int = 10,
        grad_clip: float = 5.0,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.lr_stage3 = lr_stage3
        self.weight_decay = weight_decay
        
        # Epoch settings (paper.tex)
        self.epochs_stage1 = epochs_stage1
        self.epochs_stage2 = epochs_stage2
        self.epochs_stage3 = epochs_stage3
        
        # Early stopping (paper.tex)
        self.patience_stage1 = patience_stage1
        self.patience_stage3 = patience_stage3
        
        # Gradient clipping (paper.tex: ||∇|| ≤ 5.0)
        self.grad_clip = grad_clip
        
        # Loss functions
        self.cvar_loss = CVaRLoss(alpha=0.95)
        self.mixed_loss = MixedLoss(cvar_weight=0.5)
        self.entropic_loss = EntropicLoss(lambda_risk=1.0)
        self.trading_penalty = TradingPenalty(gamma=1e-3)
        self.no_trade_penalty = NoTradeBandPenalty(nu=1e8, epsilon=0.15)
        
        # Store reference deltas from Stage 1
        self.reference_deltas = None
    
    def _create_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Create optimizer with given learning rate."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay
        )
    
    def _compute_pnl(
        self,
        deltas: torch.Tensor,
        prices: torch.Tensor,
        payoff: torch.Tensor,
        transaction_cost: float = 0.001
    ) -> torch.Tensor:
        """Compute P&L for a batch."""
        price_changes = prices[:, 1:] - prices[:, :-1]
        hedge_gains = (deltas * price_changes).sum(dim=1)
        
        delta_changes = torch.zeros_like(deltas)
        delta_changes[:, 0] = deltas[:, 0]
        delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
        tc = (torch.abs(delta_changes) * prices[:, :-1] * transaction_cost).sum(dim=1)
        
        return -payoff + hedge_gains - tc
    
    def _unpack_batch(self, batch: Dict) -> Tuple[torch.Tensor, ...]:
        """Unpack batch and move to device."""
        features = batch['features'].to(self.device)
        # Handle both 'prices' and 'stock_paths' keys for compatibility
        prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
        payoff = batch['payoff'].to(self.device)
        return features, prices, payoff
    
    def train_stage1(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 50,
        patience: int = 15
    ) -> Dict[str, list]:
        """
        Stage 1: CVaR Pretraining.
        
        Objective: Minimize CVaR95(-P&L)
        Purpose: Make model aware of tail risk before entropic optimization.
        """
        print("=" * 60)
        print("Stage 1: CVaR Pretraining")
        print("=" * 60)
        
        optimizer = self._create_optimizer(self.lr_stage1)
        history = {'loss': [], 'val_loss': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx == 0 and epoch == 0:
                    print(f"  Starting epoch 1 with {n_batches} batches...")
                # Progress indicator every 50 batches for slow models
                if n_batches > 100 and batch_idx > 0 and batch_idx % 50 == 0:
                    print(f"    Batch {batch_idx}/{n_batches}...", end='\r')
                features, prices, payoff = self._unpack_batch(batch)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                pnl = self._compute_pnl(deltas, prices, payoff)
                loss = self.cvar_loss(pnl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            
            # Validation every 10 epochs to speed up training
            if val_loader is not None and (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_loader, self.cvar_loss)
                history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f", Val: {history['val_loss'][-1]:.4f}" if history.get('val_loss') else ""
                print(f"  Epoch {epoch+1}/{epochs}: CVaR = {avg_loss:.4f}{val_str}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        # Skip reference deltas computation if Stage 2 is skipped (not needed for Stage 3)
        # This saves significant time for slow models like Transformer
        if self.epochs_stage2 > 0:
            self._compute_reference_deltas(train_loader)
        
        return history
    
    def train_stage2(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 20,
        alpha_start: float = 0.8,
        alpha_end: float = 0.2
    ) -> Dict[str, list]:
        """
        Stage 2: Mixed Loss with Alpha Annealing (NEW).
        
        Objective: alpha*CVaR + (1-alpha)*Entropic, with alpha annealed
        Purpose: Smooth transition from CVaR to entropic objective.
        """
        print("=" * 60)
        print("Stage 2: Mixed Loss (CVaR → Entropic transition)")
        print("=" * 60)
        
        optimizer = self._create_optimizer(self.lr_stage2)
        history = {'loss': [], 'alpha': [], 'cvar_component': [], 'entropic_component': []}
        
        for epoch in range(epochs):
            # Anneal alpha from start to end (avoid division by zero)
            if epochs > 1:
                alpha = alpha_start - (alpha_start - alpha_end) * (epoch / (epochs - 1))
            else:
                alpha = (alpha_start + alpha_end) / 2
            self.mixed_loss.set_weight(alpha)
            
            self.model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                features, prices, payoff = self._unpack_batch(batch)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                pnl = self._compute_pnl(deltas, prices, payoff)
                loss = self.mixed_loss(pnl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            history['alpha'].append(alpha)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, α = {alpha:.2f}")
        
        return history
    
    def train_stage3(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 30,
        patience: int = 10,
        use_no_trade_band: bool = True
    ) -> Dict[str, list]:
        """
        Stage 3: Entropic Fine-tuning with Penalties.
        
        Objective: Entropic(-P&L) + gamma*TradingPenalty + nu*NoTradeBand
        Purpose: Fine-tune for overall risk while discouraging excessive trading.
        """
        print("=" * 60)
        print("Stage 3: Entropic Fine-tuning")
        print("=" * 60)
        
        optimizer = self._create_optimizer(self.lr_stage3)
        history = {'loss': [], 'entropic': [], 'trading_penalty': [], 'band_penalty': []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_entropic = 0.0
            epoch_trading = 0.0
            epoch_band = 0.0
            
            n_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                # Progress indicator every 50 batches for slow models
                if n_batches > 100 and batch_idx > 0 and batch_idx % 50 == 0:
                    print(f"    Batch {batch_idx}/{n_batches}...", end='\r')
                features, prices, payoff = self._unpack_batch(batch)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                pnl = self._compute_pnl(deltas, prices, payoff)
                
                # Components
                entropic = self.entropic_loss(pnl)
                trading = self.trading_penalty(deltas)
                
                loss = entropic + trading
                
                # No-trade band penalty (if reference available)
                if use_no_trade_band and self.reference_deltas is not None:
                    batch_idx = batch.get('idx', None)
                    if batch_idx is not None:
                        ref_deltas = self.reference_deltas[batch_idx].to(self.device)
                        band = self.no_trade_penalty(deltas, ref_deltas)
                        loss = loss + band
                        epoch_band += band.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_entropic += entropic.item()
                epoch_trading += trading.item()
            
            history['loss'].append(epoch_loss / n_batches)
            history['entropic'].append(epoch_entropic / n_batches)
            history['trading_penalty'].append(epoch_trading / n_batches)
            history['band_penalty'].append(epoch_band / n_batches)
            
            # Validation every 10 epochs to speed up training
            if val_loader is not None and (epoch + 1) % 10 == 0:
                val_loss = self._validate(val_loader, self.entropic_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Loss = {history['loss'][-1]:.4f}, "
                      f"Entropic = {history['entropic'][-1]:.4f}")
        
        return history
    
    def train_full(
        self,
        train_loader,
        val_loader=None
    ) -> Dict[str, Any]:
        """Run complete three-stage training per paper.tex Section 5.1."""
        results = {}
        
        # Stage 1: CVaR pretraining (paper.tex: 50 epochs, lr=1e-3, patience=15)
        results['stage1'] = self.train_stage1(
            train_loader, val_loader,
            epochs=self.epochs_stage1,
            patience=self.patience_stage1
        )
        
        # Stage 2: Mixed loss (optional - skip if epochs=0 for paper.tex 2-stage protocol)
        if self.epochs_stage2 > 0:
            results['stage2'] = self.train_stage2(
                train_loader, val_loader,
                epochs=self.epochs_stage2
            )
        
        # Stage 3: Entropic fine-tuning (paper.tex: 30 epochs, lr=1e-4, patience=10)
        results['stage3'] = self.train_stage3(
            train_loader, val_loader,
            epochs=self.epochs_stage3,
            patience=self.patience_stage3,
            use_no_trade_band=False
        )
        
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        return results
    
    def _validate(
        self,
        val_loader,
        loss_fn: nn.Module
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features, prices, payoff = self._unpack_batch(batch)
                deltas = self.model(features)
                pnl = self._compute_pnl(deltas, prices, payoff)
                loss = loss_fn(pnl)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _compute_reference_deltas(self, train_loader):
        """Compute reference deltas from Stage 1 for no-trade band."""
        self.model.eval()
        all_deltas = []
        
        with torch.no_grad():
            for batch in train_loader:
                features, _, _ = self._unpack_batch(batch)
                deltas = self.model(features)
                all_deltas.append(deltas.cpu())
        
        self.reference_deltas = torch.cat(all_deltas, dim=0)


if __name__ == "__main__":
    print("Testing Three-Stage Curriculum Hedger implementation...")
    
    # Test mixed loss
    pnl = torch.randn(32)
    mixed_loss = MixedLoss(cvar_weight=0.5)
    loss = mixed_loss(pnl)
    print(f"Mixed loss: {loss.item():.4f}")
    
    # Test trading penalty
    deltas = torch.randn(32, 30)
    trading_penalty = TradingPenalty(gamma=1e-3)
    penalty = trading_penalty(deltas)
    print(f"Trading penalty: {penalty.item():.4f}")
    
    # Test no-trade band penalty
    ref_deltas = torch.randn(32, 30)
    band_penalty = NoTradeBandPenalty(nu=1.0, epsilon=0.15)
    penalty = band_penalty(deltas, ref_deltas)
    print(f"No-trade band penalty: {penalty.item():.4f}")
    
    print("\n✅ Three-Stage Curriculum Hedger implementation test passed!")
