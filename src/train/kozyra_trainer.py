"""
Two-Stage Trainer for Kozyra Models.

Implements the two-stage training procedure:
Stage 1: Frictionless pretraining with CVaR objective
Stage 2: Transaction-cost fine-tuning with band constraints
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import numpy as np
from tqdm import tqdm

try:
    from .losses import CVaRLoss, EntropicLoss, TransactionCostPenalty, NoTransactionBandPenalty
    from ..utils.logging_utils import ExperimentLogger
except ImportError:
    from train.losses import CVaRLoss, EntropicLoss, TransactionCostPenalty, NoTransactionBandPenalty
    from utils.logging_utils import ExperimentLogger


class KozyraTwoStageTrainer:
    """
    Two-stage trainer for Kozyra models.
    
    Stage 1 (frictionless pretraining):
        min_θ CVaR_α(P&L_θ)
        Freeze lower layers after convergence.
    
    Stage 2 (transaction-cost fine-tuning):
        L = Entropic(P&L) + γ * Σ|δ_{k+1} - δ_k| + ν * d(δ, H_c)
        
    Parameters:
        γ = 10^{-3} (transaction cost penalty)
        ν = 10^8 (no-transaction band penalty)
        H_c = [δ* - 0.15, δ* + 0.15] (no-transaction band)
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha_cvar: float = 0.95,
        lambda_risk: float = 1.0,
        gamma: float = 1e-3,
        nu: float = 1e8,
        band_width: float = 0.15,
        cost_multiplier: float = 0.0,
        lr_stage1: float = 0.0005,
        lr_stage2: float = 0.0001,
        device: str = 'cpu',
        logger: Optional[ExperimentLogger] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.logger = logger
        
        # Loss components
        self.cvar_loss = CVaRLoss(alpha_cvar)
        self.entropic_loss = EntropicLoss(lambda_risk)
        self.tc_penalty = TransactionCostPenalty(gamma)
        self.band_penalty = NoTransactionBandPenalty(nu, band_width)
        
        self.cost_multiplier = cost_multiplier
        self.lambda_risk = lambda_risk
        self.gamma = gamma
        self.nu = nu
        self.band_width = band_width
        
        # Optimizers for each stage
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        
        # Reference deltas from Stage 1
        self.reference_deltas = None
        
        # Training history
        self.history = {
            'stage1': {'train_loss': [], 'val_loss': []},
            'stage2': {'train_loss': [], 'val_loss': []}
        }
    
    def _compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute P&L and transaction costs."""
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        hedging_gains = torch.sum(deltas * price_changes, dim=1)
        
        deltas_ext = torch.cat([
            torch.zeros(deltas.size(0), 1, device=self.device),
            deltas,
            torch.zeros(deltas.size(0), 1, device=self.device)
        ], dim=1)
        delta_changes = torch.abs(deltas_ext[:, 1:] - deltas_ext[:, :-1])
        tc = torch.sum(delta_changes, dim=1)
        
        pnl = -payoffs + hedging_gains - self.cost_multiplier * tc
        
        return pnl, tc
    
    def _stage1_loss(self, pnl: torch.Tensor) -> torch.Tensor:
        """Stage 1 loss: CVaR."""
        return self.cvar_loss(pnl)
    
    def _stage2_loss(
        self,
        pnl: torch.Tensor,
        deltas: torch.Tensor,
        tc: torch.Tensor,
        reference_deltas: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 2 loss with all components.
        
        L = Entropic(P&L) + γ * Σ|δ_{k+1} - δ_k| + ν * d(δ, H_c)
        """
        entropic = self.entropic_loss(pnl)
        tc_pen = self.gamma * torch.mean(tc)
        band_pen = self.band_penalty(deltas, reference_deltas)
        
        total = entropic + tc_pen + band_pen
        
        components = {
            'entropic': entropic.item(),
            'tc_penalty': tc_pen.item(),
            'band_penalty': band_pen.item()
        }
        
        return total, components
    
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Stage 1: Frictionless pretraining with CVaR.
        """
        if self.logger:
            self.logger.info("Starting Stage 1: CVaR pretraining")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_stage1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(n_epochs), desc="Stage 1", disable=not verbose)
        
        for epoch in pbar:
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                
                deltas = self.model(features)
                pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
                loss = self._stage1_loss(pnl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            self.history['stage1']['train_loss'].append(train_loss)
            
            # Validation
            val_loss = self._validate_stage1(val_loader)
            self.history['stage1']['val_loss'].append(val_loss)
            
            pbar.set_postfix({'train': f"{train_loss:.4f}", 'val': f"{val_loss:.4f}"})
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if self.logger:
                    self.logger.save_checkpoint(
                        self.model, optimizer, epoch, val_loss, is_best=True
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Stage 1 early stopping at epoch {epoch}")
                    break
        
        # Store reference deltas from Stage 1
        self._compute_reference_deltas(val_loader)
        
        if self.logger:
            self.logger.info(f"Stage 1 complete. Best val loss: {best_val_loss:.4f}")
        
        return self.history['stage1']
    
    @torch.no_grad()
    def _validate_stage1(self, dataloader: DataLoader) -> float:
        """Validate Stage 1."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths'].to(self.device)
            payoffs = batch['payoff'].to(self.device)
            
            deltas = self.model(features)
            pnl, _ = self._compute_pnl(deltas, stock_paths, payoffs)
            loss = self._stage1_loss(pnl)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def _compute_reference_deltas(self, dataloader: DataLoader):
        """Compute reference deltas from Stage 1 model."""
        self.model.eval()
        all_deltas = []
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            deltas = self.model(features)
            all_deltas.append(deltas.cpu())
        
        self.reference_deltas = torch.cat(all_deltas, dim=0)
        
        if self.logger:
            self.logger.info(f"Reference deltas computed: shape {self.reference_deltas.shape}")
    
    def freeze_lower_layers(self):
        """Freeze lower layers after Stage 1."""
        if hasattr(self.model, 'freeze_lower_layers'):
            self.model.freeze_lower_layers()
        else:
            # Generic freezing - freeze all but last layer
            params = list(self.model.parameters())
            n_params = len(params)
            
            for i, param in enumerate(params):
                if i < n_params - 2:  # Keep last 2 parameter groups unfrozen
                    param.requires_grad = False
        
        if self.logger:
            n_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
            n_total = sum(1 for p in self.model.parameters())
            self.logger.info(f"Frozen {n_frozen}/{n_total} parameters for Stage 2")
    
    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 30,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Stage 2: Transaction-cost fine-tuning.
        """
        if self.reference_deltas is None:
            raise RuntimeError("Must run Stage 1 before Stage 2")
        
        if self.logger:
            self.logger.info("Starting Stage 2: Transaction-cost fine-tuning")
        
        # Freeze lower layers
        self.freeze_lower_layers()
        
        # Optimizer for unfrozen parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr_stage2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(n_epochs), desc="Stage 2", disable=not verbose)
        
        for epoch in pbar:
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            batch_idx = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                # Get corresponding reference deltas (handle variable batch sizes)
                batch_size = features.size(0)
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.reference_deltas))
                ref_deltas = self.reference_deltas[start_idx:end_idx].to(self.device)
                
                # Skip if batch sizes don't match (last batch edge case)
                if ref_deltas.size(0) != batch_size:
                    batch_idx += 1
                    continue
                
                optimizer.zero_grad()
                
                deltas = self.model(features)
                pnl, tc = self._compute_pnl(deltas, stock_paths, payoffs)
                loss, _ = self._stage2_loss(pnl, deltas, tc, ref_deltas)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
                batch_idx += 1
            
            train_loss /= n_batches
            self.history['stage2']['train_loss'].append(train_loss)
            
            # Validation
            val_loss, val_components = self._validate_stage2(val_loader)
            self.history['stage2']['val_loss'].append(val_loss)
            
            pbar.set_postfix({
                'train': f"{train_loss:.4f}",
                'val': f"{val_loss:.4f}",
                'ent': f"{val_components['entropic']:.4f}"
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if self.logger:
                    self.logger.save_checkpoint(
                        self.model, optimizer, epoch, val_loss, is_best=True
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Stage 2 early stopping at epoch {epoch}")
                    break
        
        if self.logger:
            self.logger.info(f"Stage 2 complete. Best val loss: {best_val_loss:.4f}")
        
        return self.history['stage2']
    
    @torch.no_grad()
    def _validate_stage2(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate Stage 2."""
        self.model.eval()
        total_loss = 0.0
        total_components = {'entropic': 0.0, 'tc_penalty': 0.0, 'band_penalty': 0.0}
        n_batches = 0
        batch_idx = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths'].to(self.device)
            payoffs = batch['payoff'].to(self.device)
            
            batch_size = features.size(0)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.reference_deltas))
            ref_deltas = self.reference_deltas[start_idx:end_idx].to(self.device)
            
            # Skip if batch sizes don't match
            if ref_deltas.size(0) != batch_size:
                batch_idx += 1
                continue
            
            deltas = self.model(features)
            pnl, tc = self._compute_pnl(deltas, stock_paths, payoffs)
            loss, components = self._stage2_loss(pnl, deltas, tc, ref_deltas)
            
            total_loss += loss.item()
            for key in total_components:
                total_components[key] += components[key]
            n_batches += 1
            batch_idx += 1
        
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs_stage1: int = 50,
        n_epochs_stage2: int = 30,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Complete two-stage training.
        """
        # Stage 1
        self.train_stage1(train_loader, val_loader, n_epochs_stage1, patience, verbose)
        
        # Stage 2
        self.train_stage2(train_loader, val_loader, n_epochs_stage2, patience, verbose)
        
        return self.history
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate trained model."""
        self.model.eval()
        all_pnl = []
        all_deltas = []
        all_tc = []
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths'].to(self.device)
            payoffs = batch['payoff'].to(self.device)
            
            deltas = self.model(features)
            pnl, tc = self._compute_pnl(deltas, stock_paths, payoffs)
            
            all_pnl.append(pnl.cpu().numpy())
            all_deltas.append(deltas.cpu().numpy())
            all_tc.append(tc.cpu().numpy())
        
        pnl = np.concatenate(all_pnl)
        deltas = np.concatenate(all_deltas)
        tc = np.concatenate(all_tc)
        
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
            'mean_tc': np.mean(tc),
            'total_trading_volume': np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)),
            'mean_delta': np.mean(deltas),
            'delta_std': np.mean(np.std(deltas, axis=1))
        }
        
        return metrics, pnl, deltas
