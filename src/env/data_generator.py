"""
Data Generator for Deep Hedging experiments.

Generates train/validation/test datasets as specified in Buehler et al.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict
from pathlib import Path
import pickle

from .market_env import MarketEnvironment
from .heston import HestonParams


class HedgingDataset(Dataset):
    """PyTorch Dataset for hedging data."""
    
    def __init__(
        self,
        features: np.ndarray,
        stock_paths: np.ndarray,
        payoffs: np.ndarray,
        bs_deltas: Optional[np.ndarray] = None
    ):
        """
        Args:
            features: Market features, shape (n_samples, n_steps, n_features)
            stock_paths: Stock price paths, shape (n_samples, n_steps + 1)
            payoffs: Option payoffs, shape (n_samples,)
            bs_deltas: Black-Scholes deltas, shape (n_samples, n_steps)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.stock_paths = torch.tensor(stock_paths, dtype=torch.float32)
        self.payoffs = torch.tensor(payoffs, dtype=torch.float32)
        self.bs_deltas = torch.tensor(bs_deltas, dtype=torch.float32) if bs_deltas is not None else None
        
        self.n_samples = features.shape[0]
        self.n_steps = features.shape[1]
        self.n_features = features.shape[2]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'features': self.features[idx],
            'stock_paths': self.stock_paths[idx],
            'payoff': self.payoffs[idx]
        }
        if self.bs_deltas is not None:
            item['bs_delta'] = self.bs_deltas[idx]
        return item


class DataGenerator:
    """
    Data generator for deep hedging experiments.
    
    Default dataset sizes (Buehler et al.):
    - Train: 90,000
    - Validation: 10,000  
    - Test: 100,000
    """
    
    def __init__(
        self,
        n_steps: int = 30,
        T: float = 30/365,
        S0: float = 100.0,
        K: float = 100.0,
        r: float = 0.0,
        cost_multiplier: float = 0.0,
        model_type: str = 'heston',
        heston_params: Optional[HestonParams] = None,
        bs_sigma: float = 0.2
    ):
        self.env_params = {
            'n_steps': n_steps,
            'T': T,
            'S0': S0,
            'K': K,
            'r': r,
            'cost_multiplier': cost_multiplier,
            'model_type': model_type,
            'heston_params': heston_params,
            'bs_sigma': bs_sigma
        }
        
        self.n_steps = n_steps
    
    def generate_dataset(
        self,
        n_samples: int,
        seed: int,
        compute_bs_delta: bool = True
    ) -> HedgingDataset:
        """
        Generate a dataset of hedging scenarios.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            compute_bs_delta: Whether to compute Black-Scholes deltas
        
        Returns:
            HedgingDataset
        """
        env = MarketEnvironment(**self.env_params)
        env.generate_paths(n_samples, seed=seed)
        
        features = env.get_features(normalize=True)
        stock_paths = env.stock_paths
        payoffs = env.get_payoff()
        
        bs_deltas = None
        if compute_bs_delta:
            bs_deltas = env.get_bs_delta()
        
        return HedgingDataset(features, stock_paths, payoffs, bs_deltas)
    
    def generate_train_val_test(
        self,
        n_train: int = 90000,
        n_val: int = 10000,
        n_test: int = 100000,
        base_seed: int = 42,
        compute_bs_delta: bool = True
    ) -> Tuple[HedgingDataset, HedgingDataset, HedgingDataset]:
        """
        Generate train, validation, and test datasets.
        
        Uses different seeds for each split to ensure independence.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print(f"Generating training data ({n_train} samples)...")
        train_data = self.generate_dataset(n_train, seed=base_seed, compute_bs_delta=compute_bs_delta)
        
        print(f"Generating validation data ({n_val} samples)...")
        val_data = self.generate_dataset(n_val, seed=base_seed + 1, compute_bs_delta=compute_bs_delta)
        
        print(f"Generating test data ({n_test} samples)...")
        test_data = self.generate_dataset(n_test, seed=base_seed + 2, compute_bs_delta=compute_bs_delta)
        
        return train_data, val_data, test_data
    
    def get_dataloaders(
        self,
        train_data: HedgingDataset,
        val_data: HedgingDataset,
        test_data: HedgingDataset,
        batch_size: int = 256,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train/val/test datasets."""
        
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def save_datasets(
        self,
        train_data: HedgingDataset,
        val_data: HedgingDataset,
        test_data: HedgingDataset,
        save_dir: str
    ):
        """Save datasets to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'train': {
                'features': train_data.features.numpy(),
                'stock_paths': train_data.stock_paths.numpy(),
                'payoffs': train_data.payoffs.numpy(),
                'bs_deltas': train_data.bs_deltas.numpy() if train_data.bs_deltas is not None else None
            },
            'val': {
                'features': val_data.features.numpy(),
                'stock_paths': val_data.stock_paths.numpy(),
                'payoffs': val_data.payoffs.numpy(),
                'bs_deltas': val_data.bs_deltas.numpy() if val_data.bs_deltas is not None else None
            },
            'test': {
                'features': test_data.features.numpy(),
                'stock_paths': test_data.stock_paths.numpy(),
                'payoffs': test_data.payoffs.numpy(),
                'bs_deltas': test_data.bs_deltas.numpy() if test_data.bs_deltas is not None else None
            },
            'env_params': self.env_params
        }
        
        with open(save_path / 'hedging_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Datasets saved to {save_path / 'hedging_data.pkl'}")
    
    @staticmethod
    def load_datasets(save_dir: str) -> Tuple[HedgingDataset, HedgingDataset, HedgingDataset]:
        """Load datasets from disk."""
        save_path = Path(save_dir)
        
        with open(save_path / 'hedging_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        train_data = HedgingDataset(
            data['train']['features'],
            data['train']['stock_paths'],
            data['train']['payoffs'],
            data['train']['bs_deltas']
        )
        
        val_data = HedgingDataset(
            data['val']['features'],
            data['val']['stock_paths'],
            data['val']['payoffs'],
            data['val']['bs_deltas']
        )
        
        test_data = HedgingDataset(
            data['test']['features'],
            data['test']['stock_paths'],
            data['test']['payoffs'],
            data['test']['bs_deltas']
        )
        
        return train_data, val_data, test_data


def create_sequential_features(
    features: torch.Tensor,
    prev_delta: torch.Tensor
) -> torch.Tensor:
    """
    Create features for sequential model input.
    
    Concatenates market features with previous delta position.
    
    Args:
        features: Market features (batch, n_steps, n_features)
        prev_delta: Previous delta positions (batch, n_steps, 1)
    
    Returns:
        Combined features (batch, n_steps, n_features + 1)
    """
    return torch.cat([features, prev_delta], dim=-1)
