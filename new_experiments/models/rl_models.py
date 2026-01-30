"""
Risk-Aware Reinforcement Learning for Deep Hedging

NOT vanilla MCPG. Implements:
- CVaR-PPO
- Entropic-PPO  
- TD3 with action penalty

All warm-started from supervised models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class PolicyNetwork(nn.Module):
    """Policy network for continuous action (delta increment)."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 128,
        delta_max: float = 1.5,
        max_increment: float = 0.3,
        log_std_init: float = -1.0
    ):
        super().__init__()
        self.delta_max = delta_max
        self.max_increment = max_increment
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_size),  # +1 for prev_delta
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.ones(1) * log_std_init)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mean_head.weight)
    
    def forward(self, state: torch.Tensor, prev_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, prev_delta], dim=-1)
        features = self.encoder(x)
        mean = self.max_increment * torch.tanh(self.mean_head(features))
        std = torch.clamp(self.log_std.exp(), 0.01, 0.5)
        return mean, std.expand_as(mean)
    
    def sample(self, state: torch.Tensor, prev_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state, prev_delta)
        dist = Normal(mean, std)
        increment = dist.rsample()
        log_prob = dist.log_prob(increment)
        new_delta = torch.clamp(prev_delta + increment, -self.delta_max, self.delta_max)
        return new_delta, increment, log_prob
    
    def evaluate(self, state: torch.Tensor, prev_delta: torch.Tensor, increment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state, prev_delta)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(increment)
        entropy = dist.entropy()
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value network for advantage estimation."""
    
    def __init__(self, state_dim: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
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


class QNetwork(nn.Module):
    """Q-network for TD3."""
    
    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()
        # Input: state + prev_delta + action (increment)
        self.network = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor, prev_delta: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, prev_delta, action], dim=-1)
        return self.network(x)


class CVaRPPO(nn.Module):
    """
    CVaR-aware Proximal Policy Optimization.
    
    Reward: r = -CVaR(P&L) - γ|Δδ|
    Uses tight clipping for stability.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 128,
        delta_max: float = 1.5,
        max_increment: float = 0.3,
        lr_policy: float = 1e-4,
        lr_value: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        action_penalty: float = 1e-3,
        n_epochs: int = 5,
        cvar_alpha: float = 0.95,
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
        self.cvar_alpha = cvar_alpha
        self.device = device
        self.delta_max = delta_max
        
        self.policy = PolicyNetwork(state_dim, hidden_size, delta_max, max_increment).to(device)
        self.value = ValueNetwork(state_dim, hidden_size).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Rollout buffers
        self.states = []
        self.prev_deltas = []
        self.increments = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def warm_start_from_model(self, supervised_model: nn.Module, train_loader, n_epochs: int = 10):
        """Initialize policy to mimic supervised model."""
        print("  Warm-starting CVaR-PPO from supervised model...")
        
        supervised_model.eval()
        self.policy.train()
        
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                
                with torch.no_grad():
                    target_deltas = supervised_model(features)
                
                target_increments = torch.zeros_like(target_deltas)
                target_increments[:, 0] = target_deltas[:, 0]
                target_increments[:, 1:] = target_deltas[:, 1:] - target_deltas[:, :-1]
                
                batch_size, n_steps, state_dim = features.shape
                
                loss = 0
                prev_delta = torch.zeros(batch_size, 1, device=self.device)
                
                for t in range(n_steps):
                    state = features[:, t, :]
                    mean, _ = self.policy(state, prev_delta)
                    target_inc = target_increments[:, t:t+1]
                    loss += F.mse_loss(mean, target_inc)
                    prev_delta = torch.clamp(prev_delta + target_inc, -self.delta_max, self.delta_max)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
                if n_batches >= 50:  # Limit batches per epoch
                    break
            
            if epoch % 3 == 0:
                print(f"    Epoch {epoch}: imitation loss = {total_loss/n_batches:.4f}")
    
    def select_action(self, state: torch.Tensor, prev_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_cvar_reward(self, pnl: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
        """Risk-aware reward: -CVaR - action penalty."""
        losses = -pnl
        var = torch.quantile(losses, self.cvar_alpha)
        tail = losses[losses >= var]
        cvar = tail.mean() if len(tail) > 0 else losses.mean()
        action_cost = self.action_penalty * increments.abs().sum()
        return -cvar - action_cost
    
    def compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if len(self.rewards) == 0:
            return {}
        
        if next_state is not None and next_prev_delta is not None:
            with torch.no_grad():
                next_value = self.value(next_state, next_prev_delta)
        else:
            next_value = torch.zeros(1, device=self.device)
        
        returns, advantages = self.compute_gae(next_value)
        
        states = torch.cat(self.states)
        prev_deltas = torch.cat(self.prev_deltas)
        increments = torch.cat(self.increments)
        old_log_probs = torch.cat(self.log_probs)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.n_epochs):
            new_log_probs, entropy = self.policy.evaluate(states, prev_deltas, increments)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            values = self.value(states, prev_deltas)
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # Clear buffers
        self.states = []
        self.prev_deltas = []
        self.increments = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return {
            'policy_loss': total_policy_loss / self.n_epochs,
            'value_loss': total_value_loss / self.n_epochs
        }
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Inference: generate deltas."""
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


class EntropicPPO(CVaRPPO):
    """
    PPO with entropic risk reward.
    
    Reward: r = -(1/λ)log E[exp(-λ*P&L)] - γ|Δδ|
    """
    
    def __init__(self, state_dim: int, lambda_risk: float = 1.0, **kwargs):
        super().__init__(state_dim, **kwargs)
        self.lambda_risk = lambda_risk
    
    def compute_entropic_reward(self, pnl: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
        """Entropic risk reward."""
        scaled = -self.lambda_risk * pnl
        max_val = scaled.max().detach()
        entropic = (max_val + torch.log(torch.mean(torch.exp(scaled - max_val)))) / self.lambda_risk
        action_cost = self.action_penalty * increments.abs().sum()
        return -entropic - action_cost


class TD3Hedger(nn.Module):
    """
    TD3 (Twin Delayed DDPG) for hedging.
    
    Features:
    - Twin Q-networks
    - Delayed policy updates
    - Target policy smoothing
    - Action penalty in reward
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 256,
        delta_max: float = 1.5,
        max_increment: float = 0.3,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        action_penalty: float = 1e-3,
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.action_penalty = action_penalty
        self.batch_size = batch_size
        self.device = device
        self.delta_max = delta_max
        self.max_increment = max_increment
        
        # Actor
        self.actor = PolicyNetwork(state_dim, hidden_size, delta_max, max_increment).to(device)
        self.actor_target = PolicyNetwork(state_dim, hidden_size, delta_max, max_increment).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin critics
        self.critic1 = QNetwork(state_dim, hidden_size).to(device)
        self.critic2 = QNetwork(state_dim, hidden_size).to(device)
        self.critic1_target = QNetwork(state_dim, hidden_size).to(device)
        self.critic2_target = QNetwork(state_dim, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        self.update_counter = 0
    
    def select_action(self, state: torch.Tensor, prev_delta: torch.Tensor, explore: bool = True) -> torch.Tensor:
        with torch.no_grad():
            mean, _ = self.actor(state, prev_delta)
            if explore:
                noise = torch.randn_like(mean) * self.exploration_noise
                action = mean + noise
                action = torch.clamp(action, -self.max_increment, self.max_increment)
            else:
                action = mean
        return torch.clamp(prev_delta + action, -self.delta_max, self.delta_max), action
    
    def store_transition(self, state, prev_delta, action, reward, next_state, next_prev_delta, done):
        self.buffer.append((state, prev_delta, action, reward, next_state, next_prev_delta, done))
    
    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        states, prev_deltas, actions, rewards, next_states, next_prev_deltas, dones = zip(*batch)
        
        states = torch.cat(states)
        prev_deltas = torch.cat(prev_deltas)
        actions = torch.cat(actions)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.cat(next_states)
        next_prev_deltas = torch.cat(next_prev_deltas)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Target actions with smoothing
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(actions) * 0.2,
                -self.noise_clip, self.noise_clip
            )
            next_mean, _ = self.actor_target(next_states, next_prev_deltas)
            next_actions = torch.clamp(next_mean + noise, -self.max_increment, self.max_increment)
            
            target_q1 = self.critic1_target(next_states, next_prev_deltas, next_actions)
            target_q2 = self.critic2_target(next_states, next_prev_deltas, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1 - dones) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic1(states, prev_deltas, actions)
        current_q2 = self.critic2(states, prev_deltas, actions)
        
        critic1_loss = F.mse_loss(current_q1, target)
        critic2_loss = F.mse_loss(current_q2, target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        self.update_counter += 1
        
        # Delayed policy update
        actor_loss = 0
        if self.update_counter % self.policy_delay == 0:
            mean, _ = self.actor(states, prev_deltas)
            actor_loss = -self.critic1(states, prev_deltas, mean).mean()
            actor_loss = actor_loss + self.action_penalty * mean.abs().mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            actor_loss = actor_loss.item()
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss
        }
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Inference."""
        batch_size, n_steps, _ = features.shape
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=self.device)
        
        with torch.no_grad():
            for t in range(n_steps):
                state = features[:, t, :]
                new_delta, _ = self.select_action(state, prev_delta, explore=False)
                deltas.append(new_delta.squeeze(-1))
                prev_delta = new_delta
        
        return torch.stack(deltas, dim=1)
