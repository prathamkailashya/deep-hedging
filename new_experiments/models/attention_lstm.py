"""
Attention-LSTM Hybrid for Deep Hedging

LSTM + attention over past hidden states.
Tests whether attention over memory improves tail risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base_model import BaseHedgingModel


class TemporalAttention(nn.Module):
    """
    Attention mechanism over LSTM hidden states.
    
    Computes: α_i = softmax(q^T h_i)
              H = Σ α_i h_i
    """
    
    def __init__(
        self,
        hidden_size: int,
        attention_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        # Query projection from current hidden state
        self.query = nn.Linear(hidden_size, attention_dim)
        
        # Key projection from all hidden states
        self.key = nn.Linear(hidden_size, attention_dim)
        
        # Value projection
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = attention_dim ** 0.5
    
    def forward(
        self,
        query_state: torch.Tensor,
        memory_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention over memory.
        
        Args:
            query_state: (batch, hidden_size) - current hidden state
            memory_states: (batch, memory_len, hidden_size) - past hidden states
            mask: optional causal mask
            
        Returns:
            context: (batch, hidden_size) - attended context
            weights: (batch, memory_len) - attention weights
        """
        batch, mem_len, _ = memory_states.shape
        
        # Project
        q = self.query(query_state).unsqueeze(1)  # (batch, 1, attn_dim)
        k = self.key(memory_states)  # (batch, mem_len, attn_dim)
        v = self.value(memory_states)  # (batch, mem_len, hidden)
        
        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (batch, 1, mem_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Weighted sum
        context = torch.bmm(weights, v).squeeze(1)  # (batch, hidden)
        
        return context, weights.squeeze(1)


class AttentionLSTM(BaseHedgingModel):
    """
    LSTM with attention over past hidden states.
    
    At each timestep:
    1. Run LSTM to get h_t
    2. Attend over [h_{t-L}, ..., h_{t-1}]
    3. Combine h_t and attended context
    4. Output delta
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        attention_dim: int = 32,
        memory_length: int = 10,
        combination: str = 'concat',  # 'concat', 'sum', 'gate'
        dropout: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_length = memory_length
        self.combination = combination
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = TemporalAttention(
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Combination method
        if combination == 'concat':
            self.combiner = nn.Linear(hidden_size * 2, hidden_size)
        elif combination == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        # 'sum' needs no extra parameters
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.apply(self.init_weights)
    
    def _combine(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Combine LSTM output with attention context."""
        if self.combination == 'concat':
            combined = torch.cat([h, context], dim=-1)
            return self.combiner(combined)
        elif self.combination == 'sum':
            return h + context
        elif self.combination == 'gate':
            gate = self.gate(torch.cat([h, context], dim=-1))
            return gate * h + (1 - gate) * context
        else:
            raise ValueError(f"Unknown combination: {self.combination}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with attention over LSTM memory."""
        batch, n_steps, _ = features.shape
        device = features.device
        
        # Run full LSTM
        lstm_out, _ = self.lstm(features)  # (batch, n_steps, hidden)
        
        # Store deltas
        deltas = []
        
        for t in range(n_steps):
            h_t = lstm_out[:, t, :]  # Current hidden state
            
            # Get memory (past hidden states)
            start = max(0, t - self.memory_length)
            if t > 0:
                memory = lstm_out[:, start:t, :]  # (batch, mem_len, hidden)
                
                # Attend over memory
                context, _ = self.attention(h_t, memory)
                
                # Combine
                combined = self._combine(h_t, context)
            else:
                # No memory at t=0
                combined = h_t
            
            # Output delta
            raw_delta = self.output(combined)
            deltas.append(raw_delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'memory_length': self.memory_length,
            'combination': self.combination
        })
        return config


class MultiHeadAttentionLSTM(BaseHedgingModel):
    """
    LSTM with multi-head attention over memory.
    
    Uses standard multi-head attention for richer memory interaction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_heads: int = 4,
        memory_length: int = 15,
        dropout: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        self.hidden_size = hidden_size
        self.memory_length = memory_length
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with multi-head attention over memory."""
        batch, n_steps, _ = features.shape
        
        # Run LSTM
        lstm_out, _ = self.lstm(features)
        
        deltas = []
        
        for t in range(n_steps):
            h_t = lstm_out[:, t:t+1, :]  # (batch, 1, hidden)
            
            # Get memory window
            start = max(0, t - self.memory_length)
            memory = lstm_out[:, start:t+1, :]  # Include current
            
            # Multi-head attention (query=current, key/value=memory)
            attn_out, _ = self.mha(h_t, memory, memory)
            
            # Residual + norm
            combined = self.norm(h_t + attn_out).squeeze(1)
            
            # Output
            raw_delta = self.output(combined)
            deltas.append(raw_delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)


class HierarchicalAttentionLSTM(BaseHedgingModel):
    """
    Hierarchical attention: short-term and long-term memory.
    
    Two attention mechanisms:
    1. Short-term: recent L_s steps
    2. Long-term: sampled from older history
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        short_memory: int = 5,
        long_memory: int = 15,
        attention_dim: int = 32,
        dropout: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        self.short_memory = short_memory
        self.long_memory = long_memory
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Short-term attention
        self.short_attn = TemporalAttention(hidden_size, attention_dim, dropout)
        
        # Long-term attention
        self.long_attn = TemporalAttention(hidden_size, attention_dim, dropout)
        
        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output
        self.output = nn.Linear(hidden_size, 1)
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas with hierarchical attention."""
        batch, n_steps, _ = features.shape
        
        lstm_out, _ = self.lstm(features)
        deltas = []
        
        for t in range(n_steps):
            h_t = lstm_out[:, t, :]
            
            # Short-term memory
            short_start = max(0, t - self.short_memory)
            if t > 0:
                short_mem = lstm_out[:, short_start:t, :]
                short_ctx, _ = self.short_attn(h_t, short_mem)
            else:
                short_ctx = torch.zeros_like(h_t)
            
            # Long-term memory (older than short)
            long_end = max(0, t - self.short_memory)
            long_start = max(0, long_end - self.long_memory)
            if long_end > long_start:
                long_mem = lstm_out[:, long_start:long_end, :]
                long_ctx, _ = self.long_attn(h_t, long_mem)
            else:
                long_ctx = torch.zeros_like(h_t)
            
            # Combine all
            combined = self.combiner(torch.cat([h_t, short_ctx, long_ctx], dim=-1))
            raw_delta = self.output(combined)
            deltas.append(raw_delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)
