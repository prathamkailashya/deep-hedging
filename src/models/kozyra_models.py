"""
Kozyra Models for Deep Hedging.

Implementation of RNN/LSTM models and two-stage training as described in
Kozyra (Oxford MSc thesis).

Reference: Kozyra, M. "Deep Hedging with RNNs"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np


class HedgingRNN(nn.Module):
    """
    Baseline recurrent hedging model (Kozyra-style).
    
    Architecture:
    - LSTM with hidden_size=50, num_layers=2
    - Linear output layer
    - CRITICAL: tanh bounded delta output
    
    Training setup:
    - Optimizer: Adam
    - Learning rate: 0.0005
    - Batch size: 200
    """
    
    def __init__(self, state_dim: int, hidden_size: int = 50, num_layers: int = 2,
                 delta_scale: float = 1.5):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.delta_scale = delta_scale
        
        self.rnn = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (batch, seq_len, state_dim)
            hidden: Optional initial hidden state
        
        Returns:
            delta: Hedging positions, shape (batch, seq_len)
                   BOUNDED by tanh * delta_scale
        """
        out, _ = self.rnn(x, hidden)
        raw_delta = self.fc(out)  # (batch, seq_len, 1)
        # CRITICAL: Bound delta with tanh to prevent explosion
        delta = self.delta_scale * torch.tanh(raw_delta)
        return delta.squeeze(-1)  # (batch, seq_len)


class HedgingLSTM(nn.Module):
    """
    Enhanced LSTM model with additional features.
    
    Includes:
    - Layer normalization
    - Dropout for regularization
    - Residual connections
    - CRITICAL: tanh bounded delta output
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        delta_scale: float = 1.5
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.delta_scale = delta_scale
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_size)
        
        # LSTM layers with dropout between layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size)
                for _ in range(num_layers)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input features, shape (batch, seq_len, state_dim)
        
        Returns:
            delta: Hedging positions, shape (batch, seq_len)
        """
        # Input projection
        h = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # Pass through LSTM layers with residual connections
        for i, lstm in enumerate(self.lstm_layers):
            residual = h
            h, _ = lstm(h)
            
            if self.use_layer_norm:
                h = self.layer_norms[i](h)
            
            h = self.dropout(h)
            h = h + residual  # Residual connection
        
        # Output with tanh bounding
        raw_delta = self.fc(h)  # (batch, seq_len, 1)
        # CRITICAL: Bound delta with tanh to prevent explosion
        delta = self.delta_scale * torch.tanh(raw_delta)
        return delta.squeeze(-1)  # (batch, seq_len)


class KozyraExtendedNetwork(nn.Module):
    """
    Kozyra's extended network architecture for two-stage training.
    
    Architecture layers: [1, 50, 50, 30, 1, 50, 50, 30, 1]
    
    This represents a deeper feedforward network with multiple
    bottleneck layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        layers: List[int] = None,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        if layers is None:
            # Default Kozyra architecture
            layers = [1, 50, 50, 30, 1, 50, 50, 30, 1]
        
        self.layers_config = layers
        self.use_batch_norm = use_batch_norm
        
        # Build network
        # First layer takes input_dim, subsequent layers follow the config
        network_layers = []
        
        in_dim = input_dim
        for i, out_dim in enumerate(layers):
            network_layers.append(nn.Linear(in_dim, out_dim))
            
            if i < len(layers) - 1:  # No activation on output layer
                if use_batch_norm and out_dim > 1:
                    network_layers.append(nn.BatchNorm1d(out_dim))
                network_layers.append(nn.ReLU())
            
            in_dim = out_dim
        
        self.network = nn.Sequential(*network_layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features, shape (batch, input_dim)
        
        Returns:
            delta: Hedging position, shape (batch, 1)
        """
        return self.network(x)


class KozyraTwoStageModel(nn.Module):
    """
    Kozyra's two-stage training model.
    
    Stage 1 (frictionless pretraining):
        min_θ CVaR_α(P&L_θ)
    
    Stage 2 (transaction-cost fine-tuning):
        L = Entropic(P&L) + γ * Σ|δ_{k+1} - δ_k| + ν * d(δ, H_c)
    
    Parameters:
        γ = 10^{-3} (transaction cost penalty)
        ν = 10^8 (no-transaction band penalty)
        H_c = [δ* - 0.15, δ* + 0.15] (no-transaction band)
    """
    
    def __init__(
        self,
        state_dim: int,
        n_steps: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        gamma: float = 1e-3,
        nu: float = 1e8,
        band_width: float = 0.15,
        lambda_risk: float = 1.0
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.gamma = gamma
        self.nu = nu
        self.band_width = band_width
        self.lambda_risk = lambda_risk
        
        # Base LSTM model
        self.lstm = nn.LSTM(
            input_size=state_dim + 1,  # +1 for previous delta
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Track which layers are frozen
        self.frozen_layers = set()
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def freeze_lower_layers(self):
        """Freeze LSTM layers after Stage 1 training."""
        for param in self.lstm.parameters():
            param.requires_grad = False
        self.frozen_layers.add('lstm')
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        self.frozen_layers.clear()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Market features, shape (batch, n_steps, state_dim)
        
        Returns:
            deltas: Hedging positions, shape (batch, n_steps)
        """
        batch_size = features.size(0)
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        hidden = None
        
        for k in range(self.n_steps):
            # Concatenate features with previous delta
            x = torch.cat([features[:, k:k+1, :], prev_delta.unsqueeze(1)], dim=-1)
            
            out, hidden = self.lstm(x, hidden)
            delta = self.fc(out.squeeze(1))
            
            deltas.append(delta)
            prev_delta = delta
        
        return torch.cat(deltas, dim=-1)
    
    def compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute P&L and transaction costs."""
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        hedging_gains = torch.sum(deltas * price_changes, dim=1)
        
        deltas_ext = torch.cat([
            torch.zeros(deltas.size(0), 1, device=deltas.device),
            deltas,
            torch.zeros(deltas.size(0), 1, device=deltas.device)
        ], dim=1)
        delta_changes = torch.abs(deltas_ext[:, 1:] - deltas_ext[:, :-1])
        transaction_costs = torch.sum(delta_changes, dim=1)
        
        pnl = -payoffs + hedging_gains
        
        return pnl, transaction_costs
    
    def stage1_loss(self, pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
        """
        Stage 1 loss: CVaR of losses.
        
        CVaR_α(L) = E[L | L >= VaR_α(L)]
        """
        losses = -pnl
        k = max(1, int((1 - alpha) * losses.size(0)))
        top_losses, _ = torch.topk(losses, k)
        return torch.mean(top_losses)
    
    def stage2_loss(
        self,
        pnl: torch.Tensor,
        transaction_costs: torch.Tensor,
        deltas: torch.Tensor,
        reference_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Stage 2 loss with transaction cost penalty and no-transaction band.
        
        L = Entropic(P&L) + γ * Σ|δ_{k+1} - δ_k| + ν * d(δ, H_c)
        
        Where H_c = [δ* - 0.15, δ* + 0.15]
        """
        # Entropic risk
        scaled_pnl = -self.lambda_risk * pnl
        max_val = torch.max(scaled_pnl)
        entropic = max_val + torch.log(torch.mean(torch.exp(scaled_pnl - max_val)))
        
        # Transaction cost penalty
        tc_penalty = self.gamma * torch.mean(transaction_costs)
        
        # No-transaction band penalty
        # d(δ, H_c) = max(0, |δ - δ*| - band_width)
        band_violation = torch.relu(
            torch.abs(deltas - reference_deltas) - self.band_width
        )
        band_penalty = self.nu * torch.mean(band_violation)
        
        return entropic + tc_penalty + band_penalty
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


class KozyraRNNWithAttention(nn.Module):
    """
    Enhanced Kozyra RNN with attention mechanism.
    
    Allows the model to attend to relevant historical information
    when making hedging decisions.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=state_dim + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                if 'ih' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            features: Market features, shape (batch, n_steps, state_dim)
        
        Returns:
            deltas: Hedging positions, shape (batch, n_steps)
        """
        batch_size, n_steps, _ = features.size()
        device = features.device
        
        # First pass: get LSTM representations
        prev_delta = torch.zeros(batch_size, 1, device=device)
        lstm_outputs = []
        hidden = None
        
        for k in range(n_steps):
            x = torch.cat([features[:, k:k+1, :], prev_delta.unsqueeze(1)], dim=-1)
            out, hidden = self.lstm(x, hidden)
            lstm_outputs.append(out)
            
            # Temporary delta for next step
            temp_delta = self.fc(torch.cat([out.squeeze(1), out.squeeze(1)], dim=-1))
            prev_delta = temp_delta
        
        lstm_outputs = torch.cat(lstm_outputs, dim=1)  # (batch, n_steps, hidden)
        
        # Apply self-attention
        attn_output, _ = self.attention(lstm_outputs, lstm_outputs, lstm_outputs)
        
        # Combine LSTM and attention outputs
        combined = torch.cat([lstm_outputs, attn_output], dim=-1)
        
        # Final delta predictions
        deltas = self.fc(combined).squeeze(-1)
        
        return deltas
