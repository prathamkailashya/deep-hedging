"""
Fair Training Module for Rigorous Model Comparison

Ensures identical training conditions for all models:
- Same two-stage protocol
- Same hyperparameters (except model-specific)
- Same gradient clipping
- Same early stopping
- Seed-controlled reproducibility
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import time
import json

from new_experiments.training.losses import (
    stage1_loss, stage2_loss, compute_pnl,
    cvar_loss, entropic_loss, trading_penalty
)


@dataclass
class FairTrainingConfig:
    """Identical training config for all models."""
    # Stage 1: CVaR pretraining
    stage1_epochs: int = 50
    stage1_lr: float = 0.001
    stage1_patience: int = 15
    
    # Stage 2: Entropic fine-tuning
    stage2_epochs: int = 30
    stage2_lr: float = 0.0001
    stage2_patience: int = 10
    
    # Penalties
    gamma: float = 1e-3  # Trading cost penalty
    nu: float = 1.0  # No-trade band penalty
    band_width: float = 0.15
    
    # Regularization (IDENTICAL for all models)
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    
    # Risk parameters
    lambda_risk: float = 1.0
    cvar_alpha: float = 0.95
    
    # Scheduler
    lr_patience: int = 5
    lr_factor: float = 0.5


@dataclass
class TrainingResult:
    """Complete training result with all diagnostics."""
    model_name: str
    seed: int
    
    # Training history
    stage1_train_loss: List[float] = field(default_factory=list)
    stage1_val_loss: List[float] = field(default_factory=list)
    stage2_train_loss: List[float] = field(default_factory=list)
    stage2_val_loss: List[float] = field(default_factory=list)
    
    # Test metrics
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Diagnostics
    n_parameters: int = 0
    training_time: float = 0.0
    stage1_epochs_run: int = 0
    stage2_epochs_run: int = 0
    final_lr: float = 0.0
    
    # Raw outputs for analysis
    test_pnl: Optional[np.ndarray] = None
    test_deltas: Optional[np.ndarray] = None


class FairTrainer:
    """
    Trainer ensuring absolutely fair comparison between models.
    
    Key guarantees:
    1. Identical loss functions for all models
    2. Identical optimizer settings (except lr schedule based on val loss)
    3. Identical gradient clipping
    4. Identical early stopping criteria
    5. Full seed control
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FairTrainingConfig,
        device: str = 'cpu',
        model_name: str = 'unknown'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model_name = model_name
        
        self.stage1_deltas = None
    
    def set_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_stage1(
        self,
        train_loader,
        val_loader,
        verbose: bool = True
    ) -> Tuple[List[float], List[float], int]:
        """Stage 1: CVaR pretraining."""
        cfg = self.config
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.stage1_lr,
            weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=cfg.lr_patience, factor=cfg.lr_factor
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        train_losses, val_losses = [], []
        
        pbar = tqdm(range(cfg.stage1_epochs), desc="Stage 1", disable=not verbose)
        
        for epoch in pbar:
            # Training
            self.model.train()
            epoch_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                loss = stage1_loss(deltas, stock_paths, payoffs, alpha=cfg.cvar_alpha)
                
                loss.backward()
                
                # CRITICAL: Identical gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_epoch_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = self.model(features)
                    loss = stage1_loss(deltas, stock_paths, payoffs, alpha=cfg.cvar_alpha)
                    val_epoch_losses.append(loss.item())
            
            train_loss = np.mean(epoch_losses)
            val_loss = np.mean(val_epoch_losses)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if verbose:
                pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.stage1_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Store reference deltas
        self._store_reference_deltas(train_loader)
        
        return train_losses, val_losses, epoch + 1
    
    def _store_reference_deltas(self, train_loader):
        """Store Stage 1 deltas for no-trade band."""
        self.model.eval()
        all_deltas = []
        
        with torch.no_grad():
            for batch in train_loader:
                features = batch['features'].to(self.device)
                deltas = self.model(features)
                all_deltas.append(deltas.cpu())
        
        self.stage1_deltas = torch.cat(all_deltas, dim=0)
    
    def train_stage2(
        self,
        train_loader,
        val_loader,
        verbose: bool = True
    ) -> Tuple[List[float], List[float], int, float]:
        """Stage 2: Entropic fine-tuning with penalties."""
        cfg = self.config
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.stage2_lr,
            weight_decay=cfg.weight_decay
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        train_losses, val_losses = [], []
        batch_idx = 0
        
        pbar = tqdm(range(cfg.stage2_epochs), desc="Stage 2", disable=not verbose)
        
        for epoch in pbar:
            self.model.train()
            epoch_losses = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                # Get reference deltas
                batch_size = features.size(0)
                ref_deltas = None
                if self.stage1_deltas is not None:
                    start = (batch_idx * batch_size) % len(self.stage1_deltas)
                    end = start + batch_size
                    if end <= len(self.stage1_deltas):
                        ref_deltas = self.stage1_deltas[start:end].to(self.device)
                
                optimizer.zero_grad()
                deltas = self.model(features)
                
                loss = stage2_loss(
                    deltas, stock_paths, payoffs,
                    reference_deltas=ref_deltas,
                    lambda_risk=cfg.lambda_risk,
                    gamma=cfg.gamma,
                    nu=cfg.nu,
                    band_width=cfg.band_width
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                batch_idx += 1
            
            # Validation (entropic loss only - no penalties)
            self.model.eval()
            val_epoch_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = self.model(features)
                    pnl, _ = compute_pnl(deltas, stock_paths, payoffs)
                    loss = entropic_loss(pnl, cfg.lambda_risk)
                    val_epoch_losses.append(loss.item())
            
            train_loss = np.mean(epoch_losses)
            val_loss = np.mean(val_epoch_losses)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if verbose:
                pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.stage2_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break
            
            batch_idx = 0
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        final_lr = optimizer.param_groups[0]['lr']
        return train_losses, val_losses, epoch + 1, final_lr
    
    def train(
        self,
        train_loader,
        val_loader,
        seed: int,
        verbose: bool = True
    ) -> TrainingResult:
        """Full two-stage training with complete diagnostics."""
        self.set_seed(seed)
        
        result = TrainingResult(
            model_name=self.model_name,
            seed=seed,
            n_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        )
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {self.model_name} (seed={seed})")
            print(f"Parameters: {result.n_parameters:,}")
            print(f"{'='*60}")
        
        # Stage 1
        s1_train, s1_val, s1_epochs = self.train_stage1(train_loader, val_loader, verbose)
        result.stage1_train_loss = s1_train
        result.stage1_val_loss = s1_val
        result.stage1_epochs_run = s1_epochs
        
        # Stage 2
        s2_train, s2_val, s2_epochs, final_lr = self.train_stage2(train_loader, val_loader, verbose)
        result.stage2_train_loss = s2_train
        result.stage2_val_loss = s2_val
        result.stage2_epochs_run = s2_epochs
        result.final_lr = final_lr
        
        result.training_time = time.time() - start_time
        
        return result
    
    def evaluate(self, test_loader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Comprehensive evaluation on test set."""
        self.model.eval()
        
        all_pnl, all_deltas, all_tc = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                deltas = self.model(features)
                pnl, tc = compute_pnl(deltas, stock_paths, payoffs)
                
                all_pnl.append(pnl.cpu().numpy())
                all_deltas.append(deltas.cpu().numpy())
                all_tc.append(tc.cpu().numpy())
        
        pnl = np.concatenate(all_pnl)
        deltas = np.concatenate(all_deltas)
        tc = np.concatenate(all_tc)
        
        metrics = self._compute_metrics(pnl, deltas)
        metrics['mean_tc'] = float(np.mean(tc))
        
        return metrics, pnl, deltas
    
    def _compute_metrics(self, pnl: np.ndarray, deltas: np.ndarray) -> Dict[str, float]:
        """Compute all performance metrics."""
        losses = -pnl
        
        # Trading volume
        delta_changes = np.abs(np.diff(
            np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
            axis=1
        ))
        volume = np.mean(np.sum(delta_changes, axis=1))
        
        # Entropic risk
        lambda_risk = self.config.lambda_risk
        scaled = -lambda_risk * pnl
        max_val = np.max(scaled)
        entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk
        
        return {
            'mean_pnl': float(np.mean(pnl)),
            'std_pnl': float(np.std(pnl)),
            'var_95': float(np.percentile(losses, 95)),
            'var_99': float(np.percentile(losses, 99)),
            'cvar_95': float(np.mean(losses[losses >= np.percentile(losses, 95)])),
            'cvar_99': float(np.mean(losses[losses >= np.percentile(losses, 99)])),
            'entropic_risk': float(entropic),
            'trading_volume': float(volume),
            'max_delta': float(np.max(np.abs(deltas))),
            'mean_abs_delta': float(np.mean(np.abs(deltas))),
            'delta_std': float(np.std(deltas))
        }
