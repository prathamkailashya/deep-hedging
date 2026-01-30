"""
Unified Trainer for Fair Model Comparison

All models use identical:
- Two-stage training protocol
- Gradient clipping
- Early stopping
- Evaluation metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from tqdm import tqdm
import time

from .losses import (
    stage1_loss, stage2_loss, compute_pnl, 
    cvar_loss, entropic_loss, trading_penalty
)


@dataclass
class TrainingHistory:
    """Training history for analysis."""
    stage1_train_loss: List[float]
    stage1_val_loss: List[float]
    stage2_train_loss: List[float]
    stage2_val_loss: List[float]
    best_val_cvar: float
    total_time: float


class UnifiedTrainer:
    """
    Unified trainer ensuring fair comparison.
    
    All models trained with:
    1. Stage 1: CVaR loss (frictionless hedge learning)
    2. Stage 2: Entropic + trading penalty (smooth execution)
    3. Same gradient clipping, early stopping, etc.
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_risk: float = 1.0,
        gamma: float = 1e-3,
        band_width: float = 0.15,
        grad_clip: float = 5.0,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.lambda_risk = lambda_risk
        self.gamma = gamma
        self.band_width = band_width
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.device = device
        
        self.stage1_deltas = None  # Reference for stage 2
    
    def train_stage1(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 50,
        lr: float = 0.001,
        patience: int = 15,
        verbose: bool = True
    ) -> Dict:
        """Stage 1: CVaR pretraining."""
        if verbose:
            print("\n" + "="*50)
            print("STAGE 1: CVaR Pretraining")
            print("="*50)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train': [], 'val': []}
        
        pbar = tqdm(range(n_epochs), desc="Stage 1", disable=not verbose)
        
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
                loss = stage1_loss(deltas, stock_paths, payoffs)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
                    loss = stage1_loss(deltas, stock_paths, payoffs)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train'].append(train_loss)
            history['val'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if verbose:
                pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(best_state)
        
        # Store reference deltas for stage 2
        self._store_reference_deltas(train_loader)
        
        return history
    
    def _store_reference_deltas(self, train_loader):
        """Store Stage 1 deltas as reference for no-trade band."""
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
        n_epochs: int = 30,
        lr: float = 0.0001,
        patience: int = 10,
        nu: float = 1.0,
        verbose: bool = True
    ) -> Dict:
        """Stage 2: Entropic fine-tuning with penalties."""
        if verbose:
            print("\n" + "="*50)
            print("STAGE 2: Entropic Fine-tuning")
            print("="*50)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay
        )
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train': [], 'val': []}
        
        pbar = tqdm(range(n_epochs), desc="Stage 2", disable=not verbose)
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
                if self.stage1_deltas is not None:
                    start = (batch_idx * batch_size) % len(self.stage1_deltas)
                    end = start + batch_size
                    if end > len(self.stage1_deltas):
                        start = 0
                        end = batch_size
                    ref_deltas = self.stage1_deltas[start:end].to(self.device)
                    if ref_deltas.size(0) != batch_size:
                        ref_deltas = None
                else:
                    ref_deltas = None
                
                optimizer.zero_grad()
                deltas = self.model(features)
                
                loss = stage2_loss(
                    deltas, stock_paths, payoffs,
                    reference_deltas=ref_deltas,
                    lambda_risk=self.lambda_risk,
                    gamma=self.gamma,
                    nu=nu,
                    band_width=self.band_width
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                
                train_losses.append(loss.item())
                batch_idx += 1
            
            # Validation (entropic only, no penalties)
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = self.model(features)
                    pnl, _ = compute_pnl(deltas, stock_paths, payoffs)
                    loss = entropic_loss(pnl, self.lambda_risk)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train'].append(train_loss)
            history['val'].append(val_loss)
            
            if verbose:
                pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
            
            batch_idx = 0
        
        self.model.load_state_dict(best_state)
        return history
    
    def train(
        self,
        train_loader,
        val_loader,
        stage1_epochs: int = 50,
        stage2_epochs: int = 30,
        stage1_lr: float = 0.001,
        stage2_lr: float = 0.0001,
        patience: int = 15,
        verbose: bool = True
    ) -> TrainingHistory:
        """Full two-stage training."""
        start_time = time.time()
        
        # Stage 1
        s1_history = self.train_stage1(
            train_loader, val_loader,
            n_epochs=stage1_epochs,
            lr=stage1_lr,
            patience=patience,
            verbose=verbose
        )
        
        # Stage 2
        s2_history = self.train_stage2(
            train_loader, val_loader,
            n_epochs=stage2_epochs,
            lr=stage2_lr,
            patience=patience // 2,
            verbose=verbose
        )
        
        total_time = time.time() - start_time
        
        # Get best validation CVaR
        best_val_cvar = min(s1_history['val']) if s1_history['val'] else float('inf')
        
        return TrainingHistory(
            stage1_train_loss=s1_history['train'],
            stage1_val_loss=s1_history['val'],
            stage2_train_loss=s2_history['train'],
            stage2_val_loss=s2_history['val'],
            best_val_cvar=best_val_cvar,
            total_time=total_time
        )
    
    def evaluate(self, test_loader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Comprehensive evaluation on test set."""
        self.model.eval()
        
        all_pnl = []
        all_deltas = []
        all_tc = []
        
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
        
        # Compute metrics
        metrics = compute_all_metrics(pnl, deltas, self.lambda_risk)
        metrics['mean_tc'] = float(np.mean(tc))
        
        return metrics, pnl, deltas


def compute_all_metrics(
    pnl: np.ndarray,
    deltas: np.ndarray,
    lambda_risk: float = 1.0
) -> Dict[str, float]:
    """Compute all hedging performance metrics."""
    losses = -pnl
    
    # Trading volume
    delta_changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
        axis=1
    ))
    volume = np.mean(np.sum(delta_changes, axis=1))
    
    # Entropic risk (numerically stable)
    scaled = -lambda_risk * pnl
    max_val = np.max(scaled)
    entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk
    
    # Delta smoothness
    smoothness = np.mean(np.abs(np.diff(deltas, axis=1)))
    
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
        'delta_smoothness': float(smoothness),
        'delta_std': float(np.std(deltas))
    }
