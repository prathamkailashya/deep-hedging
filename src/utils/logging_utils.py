"""
Logging and experiment tracking utilities.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Set up a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for tracking experiments, metrics, and model checkpoints."""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{self.timestamp}"
        
        # Create directories
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.figures_dir = self.run_dir / "figures"
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(
            self.run_name,
            log_file=str(self.logs_dir / "experiment.log")
        )
        
        # Metrics storage
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': []
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str = 'train'):
        """Log metrics for a given epoch and phase."""
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history[phase].append(metrics_with_epoch)
        
        metrics_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} [{phase}]: {metrics_str}")
        
        # Save metrics to file
        metrics_path = self.logs_dir / f"metrics_{phase}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history[phase], f, indent=2)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        self.logger.debug(f"Step {step} - {tag}: {value:.6f}")
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoints_dir / "latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoints_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.logger.info(f"New best model saved at epoch {epoch} with val_loss: {val_loss:.6f}")
        
        # Periodic checkpoint
        if epoch % 10 == 0:
            epoch_path = self.checkpoints_dir / f"epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_name: str = "best.pt") -> int:
        """Load model checkpoint."""
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint {checkpoint_path} not found")
            return 0
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def save_results(self, results: Dict[str, Any], filename: str = "final_results.json"):
        """Save final experiment results."""
        results_path = self.run_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
