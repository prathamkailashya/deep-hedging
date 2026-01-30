"""
Candidate 1: Wasserstein DRO Transformer (W-DRO-T)
==================================================

Grounded in: Lütkebohmert et al. 2021 "Robust deep hedging"

Key idea: Add Wasserstein Distributionally Robust Optimization (DRO) regularization
to the existing Transformer hedger to improve robustness to parameter uncertainty.

Mathematical formulation:
    L_DRO(θ) = L_entropic(θ) + ε * E[||∇_I L_entropic(θ; I)||_2]
    
where ε is the robustness radius and I are input features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.transformer import TransformerHedge
from src.train.losses import EntropicLoss, CVaRLoss


class WassersteinDROLoss(nn.Module):
    """
    Wasserstein DRO loss wrapper for any base loss function.
    
    Adds gradient-norm regularization to make the model robust to
    input perturbations within a Wasserstein ball of radius epsilon.
    
    Based on: Blanchet & Murthy 2019, "Quantifying Distributional Model Risk"
    
    Args:
        base_loss: Base loss function (e.g., EntropicLoss, CVaRLoss)
        epsilon: Robustness radius (default: 0.1)
        grad_penalty_weight: Weight for gradient penalty term
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        epsilon: float = 0.1,
        grad_penalty_weight: float = 1.0
    ):
        super().__init__()
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.grad_penalty_weight = grad_penalty_weight
    
    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        prices: torch.Tensor,
        option_payoff: torch.Tensor,
        transaction_cost: float = 0.001
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DRO loss with gradient penalty.
        
        Args:
            model: Hedging model
            inputs: Input features [B, T, d]
            prices: Price paths [B, T+1]
            option_payoff: Option payoff at maturity [B]
            transaction_cost: Proportional transaction cost
            
        Returns:
            total_loss: Base loss + DRO penalty
            metrics: Dictionary with component losses
        """
        # Enable gradient computation for inputs
        inputs_grad = inputs.detach().clone().requires_grad_(True)
        
        # Forward pass
        deltas = model(inputs_grad)
        
        # Compute P&L
        pnl = self._compute_pnl(deltas, prices, option_payoff, transaction_cost)
        
        # Base loss
        base_loss = self.base_loss(pnl)
        
        # Compute gradient w.r.t. inputs for DRO regularization
        grad_inputs = torch.autograd.grad(
            base_loss,
            inputs_grad,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient norm penalty (Wasserstein DRO dual)
        grad_norm = grad_inputs.norm(p=2, dim=-1).mean()
        dro_penalty = self.epsilon * self.grad_penalty_weight * grad_norm
        
        # Total loss
        total_loss = base_loss + dro_penalty
        
        metrics = {
            'base_loss': base_loss.detach(),
            'dro_penalty': dro_penalty.detach(),
            'grad_norm': grad_norm.detach(),
            'total_loss': total_loss.detach()
        }
        
        return total_loss, metrics
    
    def _compute_pnl(
        self,
        deltas: torch.Tensor,
        prices: torch.Tensor,
        option_payoff: torch.Tensor,
        transaction_cost: float
    ) -> torch.Tensor:
        """Compute hedging P&L."""
        B, T = deltas.shape
        
        # Hedging gains: sum of delta * (S_{t+1} - S_t)
        price_changes = prices[:, 1:] - prices[:, :-1]  # [B, T]
        hedge_gains = (deltas * price_changes).sum(dim=1)  # [B]
        
        # Transaction costs: sum of |delta_t - delta_{t-1}| * S_t * cost
        delta_changes = torch.zeros_like(deltas)
        delta_changes[:, 0] = deltas[:, 0]  # Initial position
        delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
        tc = (torch.abs(delta_changes) * prices[:, :-1] * transaction_cost).sum(dim=1)
        
        # P&L = -option_payoff + hedge_gains - transaction_costs
        pnl = -option_payoff + hedge_gains - tc
        
        return pnl


class WDROTransformerHedger(nn.Module):
    """
    Transformer Hedger with Wasserstein DRO training.
    
    Wraps the existing TransformerHedge model and adds DRO training capability.
    
    Args:
        input_dim: Input feature dimension
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        delta_max: Maximum delta bound
        epsilon: DRO robustness radius
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        delta_max: float = 1.5,
        epsilon: float = 0.1
    ):
        super().__init__()
        
        # Base transformer model
        self.transformer = TransformerHedge(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.epsilon = epsilon
        self.delta_max = delta_max
        
        # DRO loss wrapper
        self.dro_loss = WassersteinDROLoss(
            base_loss=EntropicLoss(lambda_risk=1.0),
            epsilon=epsilon
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer."""
        return self.transformer(x)
    
    def compute_dro_loss(
        self,
        inputs: torch.Tensor,
        prices: torch.Tensor,
        option_payoff: torch.Tensor,
        transaction_cost: float = 0.001
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute DRO-regularized loss."""
        return self.dro_loss(
            self.transformer,
            inputs,
            prices,
            option_payoff,
            transaction_cost
        )


class WDROTrainer:
    """
    Three-phase trainer for W-DRO-T model.
    
    Training protocol:
        Phase 1: CVaR pretraining (50 epochs) - standard
        Phase 2: Entropic + DRO (30 epochs) - ε annealed 0→0.1
        Phase 3: Stress testing (10 epochs) - ε=0.2 on adversarial params
    """
    
    def __init__(
        self,
        model: WDROTransformerHedger,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss functions for different phases
        self.cvar_loss = CVaRLoss(alpha=0.95)
        self.entropic_loss = EntropicLoss(lambda_risk=1.0)
    
    def train_phase1(
        self,
        train_loader,
        epochs: int = 50,
        patience: int = 15
    ) -> Dict[str, list]:
        """Phase 1: CVaR pretraining."""
        print("Phase 1: CVaR Pretraining")
        history = {'loss': [], 'cvar': []}
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx == 0 and epoch == 0:
                    print(f"  Starting epoch 1 with {n_batches} batches...")
                # Progress indicator every 50 batches for slow models
                if n_batches > 100 and batch_idx > 0 and batch_idx % 50 == 0:
                    print(f"    Batch {batch_idx}/{n_batches}...", end='\r')
                inputs, prices, payoff = self._unpack_batch(batch)
                
                self.optimizer.zero_grad()
                deltas = self.model(inputs)
                pnl = self._compute_pnl(deltas, prices, payoff)
                loss = self.cvar_loss(pnl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: CVaR Loss = {avg_loss:.4f}")
        
        return history
    
    def train_phase2(
        self,
        train_loader,
        epochs: int = 30,
        epsilon_start: float = 0.0,
        epsilon_end: float = 0.1
    ) -> Dict[str, list]:
        """Phase 2: Entropic + DRO with epsilon annealing."""
        print("Phase 2: Entropic + DRO (epsilon annealing)")
        history = {'loss': [], 'dro_penalty': [], 'epsilon': []}
        
        for epoch in range(epochs):
            # Anneal epsilon
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (epoch / epochs)
            self.model.dro_loss.epsilon = epsilon
            
            epoch_loss = 0.0
            epoch_dro = 0.0
            
            for batch in train_loader:
                inputs, prices, payoff = self._unpack_batch(batch)
                
                self.optimizer.zero_grad()
                loss, metrics = self.model.compute_dro_loss(inputs, prices, payoff)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                epoch_loss += metrics['total_loss'].item()
                epoch_dro += metrics['dro_penalty'].item()
            
            history['loss'].append(epoch_loss / len(train_loader))
            history['dro_penalty'].append(epoch_dro / len(train_loader))
            history['epsilon'].append(epsilon)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss = {history['loss'][-1]:.4f}, "
                      f"DRO = {history['dro_penalty'][-1]:.4f}, ε = {epsilon:.3f}")
        
        return history
    
    def train_phase3(
        self,
        train_loader,
        epochs: int = 10,
        epsilon: float = 0.2
    ) -> Dict[str, list]:
        """Phase 3: Stress testing with high epsilon."""
        print("Phase 3: Stress Testing (high epsilon)")
        self.model.dro_loss.epsilon = epsilon
        history = {'loss': [], 'dro_penalty': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_dro = 0.0
            
            for batch in train_loader:
                inputs, prices, payoff = self._unpack_batch(batch)
                
                self.optimizer.zero_grad()
                loss, metrics = self.model.compute_dro_loss(inputs, prices, payoff)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                epoch_loss += metrics['total_loss'].item()
                epoch_dro += metrics['dro_penalty'].item()
            
            history['loss'].append(epoch_loss / len(train_loader))
            history['dro_penalty'].append(epoch_dro / len(train_loader))
            
            print(f"  Epoch {epoch+1}: Loss = {history['loss'][-1]:.4f}, "
                  f"DRO = {history['dro_penalty'][-1]:.4f}")
        
        return history
    
    def train_full(self, train_loader) -> Dict[str, Any]:
        """Run all three training phases."""
        results = {}
        results['phase1'] = self.train_phase1(train_loader)
        results['phase2'] = self.train_phase2(train_loader)
        results['phase3'] = self.train_phase3(train_loader)
        return results
    
    def train(self, train_loader, val_loader=None, epochs=80):
        """Simple training interface for compatibility."""
        # Split epochs: 50 phase1, 30 phase2 (matching paper.tex 2-stage)
        phase1_epochs = min(50, epochs)
        phase2_epochs = max(0, epochs - 50)
        
        self.train_phase1(train_loader, epochs=phase1_epochs)
        if phase2_epochs > 0:
            self.train_phase2(train_loader, epochs=phase2_epochs)
    
    def _unpack_batch(self, batch):
        """Unpack batch and move to device."""
        inputs = batch['features'].to(self.device)
        # Handle both 'prices' and 'stock_paths' keys for compatibility
        prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
        payoff = batch['payoff'].to(self.device)
        return inputs, prices, payoff
    
    def _compute_pnl(self, deltas, prices, payoff, tc=0.001):
        """Compute P&L for a batch."""
        price_changes = prices[:, 1:] - prices[:, :-1]
        hedge_gains = (deltas * price_changes).sum(dim=1)
        
        delta_changes = torch.zeros_like(deltas)
        delta_changes[:, 0] = deltas[:, 0]
        delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
        transaction_costs = (torch.abs(delta_changes) * prices[:, :-1] * tc).sum(dim=1)
        
        return -payoff + hedge_gains - transaction_costs


if __name__ == "__main__":
    # Test W-DRO-T implementation
    print("Testing W-DRO-T implementation...")
    
    # Create model
    model = WDROTransformerHedger(
        input_dim=5,
        d_model=64,
        n_heads=4,
        n_layers=3,
        epsilon=0.1
    )
    
    # Create dummy data
    B, T, d = 32, 30, 5
    inputs = torch.randn(B, T, d)
    prices = 100 + torch.cumsum(torch.randn(B, T+1) * 0.5, dim=1)
    payoff = F.relu(prices[:, -1] - 100)  # Call option payoff
    
    # Test forward pass
    deltas = model(inputs)
    print(f"Deltas shape: {deltas.shape}")
    print(f"Delta range: [{deltas.min():.3f}, {deltas.max():.3f}]")
    
    # Test DRO loss
    loss, metrics = model.compute_dro_loss(inputs, prices, payoff)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Base loss: {metrics['base_loss'].item():.4f}")
    print(f"DRO penalty: {metrics['dro_penalty'].item():.4f}")
    print(f"Grad norm: {metrics['grad_norm'].item():.4f}")
    
    print("\n✅ W-DRO-T implementation test passed!")
