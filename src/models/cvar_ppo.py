"""
CVaR-PPO: Risk-Aware Proximal Policy Optimization

Implements PPO with CVaR/Entropic risk objectives for hedging.
Key features:
- Warm-start from supervised model
- Action penalty on |Δδ| for smooth trading
- Risk-aware reward shaping
- Strict gradient clipping for stability

References:
- Schulman et al. "Proximal Policy Optimization Algorithms"
- Tamar et al. "Policy Gradients with Variance Related Risk Criteria"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
from collections import deque


class RiskAwarePolicy(nn.Module):
    """
    Policy network for risk-aware hedging.
    
    Outputs delta increments (not direct deltas) for smooth control.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        delta_max: float = 1.5,
        max_increment: float = 0.2,
        log_std_init: float = -1.0
    ):
        super().__init__()
        self.delta_max = delta_max
        self.max_increment = max_increment
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for prev_delta
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean head for delta increment
        self.mean_head = nn.Linear(hidden_dim, 1)
        
        # Learnable log std (bounded)
        self.log_std = nn.Parameter(torch.ones(1) * log_std_init)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mean_head.weight)
    
    def forward(self, state: torch.Tensor, prev_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std of increment distribution."""
        x = torch.cat([state, prev_delta], dim=-1)
        features = self.encoder(x)
        
        mean = self.max_increment * torch.tanh(self.mean_head(features))
        std = torch.clamp(self.log_std.exp(), 0.01, 0.5)
        
        return mean, std.expand_as(mean)
    
    def sample(
        self, 
        state: torch.Tensor, 
        prev_delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, std = self.forward(state, prev_delta)
        dist = Normal(mean, std)
        
        # Sample increment
        increment = dist.rsample()
        log_prob = dist.log_prob(increment)
        
        # Compute new delta
        new_delta = torch.clamp(prev_delta + increment, -self.delta_max, self.delta_max)
        
        return new_delta, increment, log_prob
    
    def evaluate(
        self,
        state: torch.Tensor,
        prev_delta: torch.Tensor,
        increment: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy."""
        mean, std = self.forward(state, prev_delta)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(increment)
        entropy = dist.entropy()
        
        return log_prob, entropy


class RiskAwareValue(nn.Module):
    """Value network estimating expected risk."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, prev_delta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, prev_delta], dim=-1)
        return self.network(x)


class CVaRPPO(nn.Module):
    """
    CVaR-aware PPO for deep hedging.
    
    Key modifications from vanilla PPO:
    1. Risk-aware reward: r = -CVaR(P&L) - γ|Δδ|
    2. Warm-start from supervised model
    3. Strict clipping and gradient bounds
    4. Short horizon updates
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        delta_max: float = 1.5,
        max_increment: float = 0.2,
        lr_policy: float = 1e-4,
        lr_value: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,  # Tighter than standard 0.2
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        action_penalty: float = 1e-3,
        n_epochs: int = 5,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.action_penalty = action_penalty
        self.n_epochs = n_epochs
        self.device = device
        self.delta_max = delta_max
        
        # Networks
        self.policy = RiskAwarePolicy(
            state_dim, hidden_dim, delta_max, max_increment
        ).to(device)
        self.value = RiskAwareValue(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Rollout storage
        self.states = []
        self.prev_deltas = []
        self.increments = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def warm_start_from_model(self, supervised_model: nn.Module, train_loader):
        """
        Initialize policy to mimic supervised model.
        
        This is crucial - don't learn hedging from scratch!
        """
        print("  Warm-starting from supervised model...")
        
        supervised_model.eval()
        self.policy.train()
        
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        for epoch in range(10):
            total_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                
                with torch.no_grad():
                    target_deltas = supervised_model(features)
                
                # Compute target increments
                target_increments = torch.zeros_like(target_deltas)
                target_increments[:, 0] = target_deltas[:, 0]
                target_increments[:, 1:] = target_deltas[:, 1:] - target_deltas[:, :-1]
                
                # Imitation loss
                batch_size, n_steps, state_dim = features.shape
                
                loss = 0
                prev_delta = torch.zeros(batch_size, 1, device=self.device)
                
                for t in range(n_steps):
                    state = features[:, t, :]
                    mean, _ = self.policy(state, prev_delta)
                    
                    target_inc = target_increments[:, t:t+1]
                    loss += F.mse_loss(mean, target_inc)
                    
                    prev_delta = torch.clamp(
                        prev_delta + target_inc, 
                        -self.delta_max, self.delta_max
                    )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if epoch % 3 == 0:
                print(f"    Epoch {epoch}: imitation loss = {total_loss/n_batches:.4f}")
    
    def select_action(
        self, 
        state: torch.Tensor, 
        prev_delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action and store for update."""
        with torch.no_grad():
            value = self.value(state, prev_delta)
        
        new_delta, increment, log_prob = self.policy.sample(state, prev_delta)
        
        self.states.append(state.detach())
        self.prev_deltas.append(prev_delta.detach())
        self.increments.append(increment.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        
        return new_delta, increment
    
    def store_reward(self, reward: float, done: bool = False):
        """Store step reward."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_risk_reward(
        self,
        pnl: torch.Tensor,
        increments: torch.Tensor,
        alpha: float = 0.95
    ) -> torch.Tensor:
        """
        Compute risk-aware reward.
        
        reward = -CVaR(P&L) - γ * Σ|Δδ|
        """
        # CVaR component
        losses = -pnl
        var = torch.quantile(losses, alpha)
        cvar = losses[losses >= var].mean() if (losses >= var).any() else losses.mean()
        
        # Action penalty
        action_cost = self.action_penalty * increments.abs().sum()
        
        return -cvar - action_cost
    
    def compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        values = self.values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        returns = torch.cat(returns)
        advantages = torch.cat(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, next_state: Optional[torch.Tensor] = None, 
               next_prev_delta: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """PPO update with risk-aware objectives."""
        if len(self.rewards) == 0:
            return {}
        
        # Compute final value
        if next_state is not None and next_prev_delta is not None:
            with torch.no_grad():
                next_value = self.value(next_state, next_prev_delta)
        else:
            next_value = torch.zeros(1, device=self.device)
        
        returns, advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.cat(self.states)
        prev_deltas = torch.cat(self.prev_deltas)
        increments = torch.cat(self.increments)
        old_log_probs = torch.cat(self.log_probs)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.n_epochs):
            # Policy loss with clipping
            new_log_probs, entropy = self.policy.evaluate(states, prev_deltas, increments)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value(states, prev_deltas)
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear storage
        self.states = []
        self.prev_deltas = []
        self.increments = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return {
            'policy_loss': total_policy_loss / self.n_epochs,
            'value_loss': total_value_loss / self.n_epochs,
            'entropy': total_entropy / self.n_epochs
        }
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate hedging deltas (inference mode)."""
        batch_size, n_steps, _ = features.shape
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=self.device)
        
        with torch.no_grad():
            for t in range(n_steps):
                state = features[:, t, :]
                mean, _ = self.policy(state, prev_delta)
                new_delta = torch.clamp(prev_delta + mean, -self.delta_max, self.delta_max)
                deltas.append(new_delta.squeeze(-1))
                prev_delta = new_delta
        
        return torch.stack(deltas, dim=1)


class CVaRPPOTrainer:
    """
    Trainer for CVaR-PPO with hedging environment.
    """
    
    def __init__(
        self,
        model: CVaRPPO,
        lambda_risk: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model
        self.lambda_risk = lambda_risk
        self.device = device
    
    def train(
        self,
        train_loader,
        n_episodes: int = 200,
        rollout_steps: int = 30,
        verbose: bool = True
    ) -> Dict:
        """
        Train CVaR-PPO on hedging environment.
        """
        history = {'rewards': [], 'policy_loss': [], 'value_loss': []}
        
        from tqdm import tqdm
        pbar = tqdm(range(n_episodes), desc="CVaR-PPO") if verbose else range(n_episodes)
        
        for episode in pbar:
            episode_rewards = []
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                batch_size, n_steps, state_dim = features.shape
                
                # Rollout
                prev_delta = torch.zeros(batch_size, 1, device=self.device)
                all_deltas = []
                all_increments = []
                
                for t in range(n_steps):
                    state = features[:, t, :]
                    new_delta, increment = self.model.select_action(state, prev_delta)
                    
                    all_deltas.append(new_delta)
                    all_increments.append(increment)
                    prev_delta = new_delta
                    
                    # Step reward (intermediate)
                    if t < n_steps - 1:
                        step_reward = -self.model.action_penalty * increment.abs().mean().item()
                        self.model.store_reward(step_reward, done=False)
                
                # Final P&L reward
                deltas = torch.stack([d.squeeze(-1) for d in all_deltas], dim=1)
                increments = torch.stack([i.squeeze(-1) for i in all_increments], dim=1)
                
                price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
                hedging_gains = torch.sum(deltas * price_changes, dim=1)
                pnl = -payoffs + hedging_gains
                
                # Risk-aware terminal reward
                final_reward = self.model.compute_risk_reward(pnl, increments).item()
                self.model.store_reward(final_reward, done=True)
                
                episode_rewards.append(final_reward)
                
                # Update
                metrics = self.model.update(
                    next_state=features[:, -1, :],
                    next_prev_delta=prev_delta
                )
                
                if metrics:
                    history['policy_loss'].append(metrics['policy_loss'])
                    history['value_loss'].append(metrics['value_loss'])
                
                break  # One batch per episode
            
            avg_reward = np.mean(episode_rewards)
            history['rewards'].append(avg_reward)
            
            if verbose and episode % 20 == 0:
                pbar.set_postfix({'reward': f"{avg_reward:.4f}"})
        
        return history
    
    def evaluate(self, test_loader) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Evaluate trained model."""
        self.model.eval()
        
        all_pnl = []
        all_deltas = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                deltas = self.model(features)
                
                price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
                hedging_gains = torch.sum(deltas * price_changes, dim=1)
                pnl = -payoffs + hedging_gains
                
                all_pnl.append(pnl.cpu().numpy())
                all_deltas.append(deltas.cpu().numpy())
        
        pnl = np.concatenate(all_pnl)
        deltas = np.concatenate(all_deltas)
        
        # Compute metrics
        losses = -pnl
        delta_changes = np.abs(np.diff(
            np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
            axis=1
        ))
        volume = np.mean(np.sum(delta_changes, axis=1))
        
        scaled = -self.lambda_risk * pnl
        max_val = np.max(scaled)
        entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / self.lambda_risk
        
        metrics = {
            'mean_pnl': np.mean(pnl),
            'std_pnl': np.std(pnl),
            'var_95': np.percentile(losses, 95),
            'var_99': np.percentile(losses, 99),
            'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
            'entropic_risk': entropic,
            'trading_volume': volume,
            'max_delta': np.max(np.abs(deltas))
        }
        
        return metrics, pnl, deltas
