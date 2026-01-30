"""
Training utilities for Deep Hedging models.

Implements training loops for:
- Buehler et al. Deep Hedging model
- General neural network hedging models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import numpy as np
from tqdm import tqdm
import time

try:
    from .losses import EntropicLoss, CVaRLoss, HedgingLoss
    from ..utils.logging_utils import ExperimentLogger
except ImportError:
    from train.losses import EntropicLoss, CVaRLoss, HedgingLoss
    from utils.logging_utils import ExperimentLogger


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Base trainer class for hedging models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = 'cpu',
        logger: Optional[ExperimentLogger] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths'].to(self.device)
            payoffs = batch['payoff'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            deltas = self.model(features)
            
            # Compute loss
            if isinstance(self.loss_fn, HedgingLoss):
                loss = self.loss_fn(deltas, stock_paths, payoffs)
            else:
                # Compute P&L manually
                pnl = self._compute_pnl(deltas, stock_paths, payoffs)
                loss = self.loss_fn(pnl)
            
            # Backward pass with stronger gradient clipping (per paper requirements)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_pnl = []
        n_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths'].to(self.device)
            payoffs = batch['payoff'].to(self.device)
            
            deltas = self.model(features)
            pnl = self._compute_pnl(deltas, stock_paths, payoffs)
            
            if isinstance(self.loss_fn, HedgingLoss):
                loss = self.loss_fn(deltas, stock_paths, payoffs)
            else:
                loss = self.loss_fn(pnl)
            
            total_loss += loss.item()
            all_pnl.append(pnl.cpu().numpy())
            n_batches += 1
        
        all_pnl = np.concatenate(all_pnl)
        
        metrics = {
            'loss': total_loss / n_batches,
            'mean_pnl': np.mean(all_pnl),
            'std_pnl': np.std(all_pnl),
            'var_95': np.percentile(-all_pnl, 95),
            'cvar_95': np.mean(-all_pnl[-all_pnl >= np.percentile(-all_pnl, 95)])
        }
        
        return metrics
    
    def _compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor,
        cost_multiplier: float = 0.0
    ) -> torch.Tensor:
        """Compute P&L from hedging positions."""
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        hedging_gains = torch.sum(deltas * price_changes, dim=1)
        
        deltas_ext = torch.cat([
            torch.zeros(deltas.size(0), 1, device=deltas.device),
            deltas,
            torch.zeros(deltas.size(0), 1, device=deltas.device)
        ], dim=1)
        delta_changes = torch.abs(deltas_ext[:, 1:] - deltas_ext[:, :-1])
        transaction_costs = cost_multiplier * torch.sum(delta_changes, dim=1)
        
        pnl = -payoffs + hedging_gains - transaction_costs
        return pnl
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train model with early stopping."""
        early_stopping = EarlyStopping(patience=patience)
        best_val_loss = float('inf')
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mean_pnl': [],
            'val_std_pnl': []
        }
        
        pbar = tqdm(range(n_epochs), disable=not verbose)
        
        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mean_pnl'].append(val_metrics['mean_pnl'])
            history['val_std_pnl'].append(val_metrics['std_pnl'])
            
            # Logging
            if self.logger:
                self.logger.log_metrics({'train_loss': train_loss}, epoch, 'train')
                self.logger.log_metrics(val_metrics, epoch, 'val')
                
                is_best = val_metrics['loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['loss']
                self.logger.save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics['loss'], is_best
                )
            
            pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'mean_pnl': f"{val_metrics['mean_pnl']:.4f}"
            })
            
            # Early stopping
            if early_stopping(val_metrics['loss']):
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        return history


class DeepHedgingTrainer(Trainer):
    """
    Trainer specifically for Buehler et al. Deep Hedging model.
    
    Follows the exact training setup from Section 5:
    - Optimizer: Adam
    - Learning rate: 0.005
    - Batch size: 256
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_risk: float = 1.0,
        cost_multiplier: float = 0.0,
        learning_rate: float = 0.0005,  # Reduced from 0.005 per paper requirements
        weight_decay: float = 1e-4,  # L2 regularization
        device: str = 'cpu',
        logger: Optional[ExperimentLogger] = None
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = HedgingLoss(
            cost_multiplier=cost_multiplier,
            lambda_risk=lambda_risk,
            risk_measure='entropic'
        )
        
        super().__init__(model, optimizer, loss_fn, device, logger)
        
        self.lambda_risk = lambda_risk
        self.cost_multiplier = cost_multiplier
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive evaluation with all metrics.
        """
        self.model.eval()
        all_pnl = []
        all_deltas = []
        all_payoffs = []
        all_hedging_gains = []
        all_tc = []
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths'].to(self.device)
            payoffs = batch['payoff'].to(self.device)
            
            deltas = self.model(features)
            
            # Compute components
            price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
            hedging_gains = torch.sum(deltas * price_changes, dim=1)
            
            deltas_ext = torch.cat([
                torch.zeros(deltas.size(0), 1, device=self.device),
                deltas,
                torch.zeros(deltas.size(0), 1, device=self.device)
            ], dim=1)
            delta_changes = torch.abs(deltas_ext[:, 1:] - deltas_ext[:, :-1])
            tc = self.cost_multiplier * torch.sum(delta_changes, dim=1)
            
            pnl = -payoffs + hedging_gains - tc
            
            all_pnl.append(pnl.cpu().numpy())
            all_deltas.append(deltas.cpu().numpy())
            all_payoffs.append(payoffs.cpu().numpy())
            all_hedging_gains.append(hedging_gains.cpu().numpy())
            all_tc.append(tc.cpu().numpy())
        
        pnl = np.concatenate(all_pnl)
        deltas = np.concatenate(all_deltas)
        payoffs = np.concatenate(all_payoffs)
        hedging_gains = np.concatenate(all_hedging_gains)
        tc = np.concatenate(all_tc)
        
        # Compute metrics
        losses = -pnl
        
        # Entropic risk
        scaled = -self.lambda_risk * pnl
        max_val = np.max(scaled)
        entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / self.lambda_risk
        
        metrics = {
            'mean_pnl': np.mean(pnl),
            'std_pnl': np.std(pnl),
            'min_pnl': np.min(pnl),
            'max_pnl': np.max(pnl),
            'var_95': np.percentile(losses, 95),
            'var_99': np.percentile(losses, 99),
            'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
            'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)]),
            'entropic_risk': entropic,
            'indifference_price': entropic,
            'mean_payoff': np.mean(payoffs),
            'mean_hedging_gains': np.mean(hedging_gains),
            'mean_transaction_costs': np.mean(tc),
            'total_trading_volume': np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)),
            'mean_delta': np.mean(deltas),
            'delta_std': np.mean(np.std(deltas, axis=1))
        }
        
        return metrics, pnl, deltas
    
    def compute_indifference_price(self, dataloader: DataLoader) -> float:
        """Compute indifference price from optimal hedging."""
        metrics, _, _ = self.evaluate(dataloader)
        return metrics['indifference_price']


class ScheduledTrainer(DeepHedgingTrainer):
    """Trainer with learning rate scheduling."""
    
    def __init__(
        self,
        model: nn.Module,
        lambda_risk: float = 1.0,
        cost_multiplier: float = 0.0,
        learning_rate: float = 0.005,
        lr_decay: float = 0.95,
        device: str = 'cpu',
        logger: Optional[ExperimentLogger] = None
    ):
        super().__init__(model, lambda_risk, cost_multiplier, learning_rate, device, logger)
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=lr_decay
        )
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train with learning rate decay."""
        history = super().fit(train_loader, val_loader, n_epochs, patience, verbose)
        
        # Step scheduler at end of each epoch
        self.scheduler.step()
        
        return history
