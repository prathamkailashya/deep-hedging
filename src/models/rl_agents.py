"""
Reinforcement Learning Agents for Deep Hedging.

Implements RL-based hedging strategies:
- MCPG (Monte Carlo Policy Gradient)
- PPO (Proximal Policy Optimization)
- DDPG/TD3 (Deep Deterministic Policy Gradient / Twin Delayed DDPG)

References:
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- Lillicrap et al. (2015) "Continuous Control with Deep Reinforcement Learning"
- Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """Policy network for continuous action space."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 64,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Initialize output layers with smaller weights
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and log_std of action distribution."""
        x = self.shared(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of given action."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value/Critic network."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class QNetwork(nn.Module):
    """Q-network for actor-critic methods."""
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class MCPGHedge(nn.Module):
    """
    Monte Carlo Policy Gradient (REINFORCE) for hedging.
    
    Uses the policy gradient theorem with Monte Carlo returns:
    ∇J(θ) = E[∇log π(a|s) * R]
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01
    ):
        super().__init__()
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.policy = PolicyNetwork(state_dim + 1, 1, hidden_dim)  # +1 for prev_delta
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for episode
        self.log_probs = []
        self.rewards = []
        self.entropies = []
    
    def select_action(self, state: torch.Tensor, prev_delta: torch.Tensor) -> torch.Tensor:
        """Select action using current policy."""
        x = torch.cat([state, prev_delta], dim=-1)
        action, log_prob = self.policy.sample(x)
        
        # Store for training
        _, entropy = self.policy.evaluate(x, action)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        
        return action
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self) -> float:
        """Update policy using collected episode."""
        returns = self.compute_returns()
        
        policy_loss = 0
        entropy_loss = 0
        
        for log_prob, entropy, R in zip(self.log_probs, self.entropies, returns):
            policy_loss -= log_prob * R
            entropy_loss -= entropy
        
        loss = policy_loss.mean() + self.entropy_coef * entropy_loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear storage
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        
        return loss.item()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate hedging strategy for entire sequence."""
        batch_size, n_steps, state_dim = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        with torch.no_grad():
            for k in range(n_steps):
                state = features[:, k, :]
                x = torch.cat([state, prev_delta], dim=-1)
                
                mean, _ = self.policy.forward(x)
                delta = mean  # Use mean for inference
                
                deltas.append(delta.squeeze(-1))
                prev_delta = delta
        
        return torch.stack(deltas, dim=1)


class PPOHedge(nn.Module):
    """
    Proximal Policy Optimization for hedging.
    
    Uses clipped surrogate objective:
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 10
    ):
        super().__init__()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        
        # Networks
        self.policy = PolicyNetwork(state_dim + 1, 1, hidden_dim)
        self.value = ValueNetwork(state_dim + 1, hidden_dim)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
        # Storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state: torch.Tensor, prev_delta: torch.Tensor) -> torch.Tensor:
        """Select action and store transition data."""
        x = torch.cat([state, prev_delta], dim=-1)
        
        action, log_prob = self.policy.sample(x)
        value = self.value(x)
        
        self.states.append(x.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        
        return action
    
    def store_reward(self, reward: float, done: bool = False):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
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
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self, next_state: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Update policy and value networks."""
        if next_state is not None:
            next_value = self.value(next_state)
        else:
            next_value = torch.zeros(1)
        
        returns, advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.n_epochs):
            # Policy loss
            new_log_probs, entropy = self.policy.evaluate(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = self.value(states)
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
        self.actions = []
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
        """Generate hedging strategy."""
        batch_size, n_steps, state_dim = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        with torch.no_grad():
            for k in range(n_steps):
                state = features[:, k, :]
                x = torch.cat([state, prev_delta], dim=-1)
                
                mean, _ = self.policy.forward(x)
                delta = mean
                
                deltas.append(delta.squeeze(-1))
                prev_delta = delta
        
        return torch.stack(deltas, dim=1)


class DDPGHedge(nn.Module):
    """
    Deep Deterministic Policy Gradient for hedging.
    
    Uses deterministic policy with off-policy learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        buffer_size: int = 100000
    ):
        super().__init__()
        
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        
        # Actor networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Bound actions
        )
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = QNetwork(state_dim + 1, 1, hidden_dim)
        self.critic_target = QNetwork(state_dim + 1, 1, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: torch.Tensor, prev_delta: torch.Tensor, 
                     add_noise: bool = True) -> torch.Tensor:
        """Select action with optional exploration noise."""
        x = torch.cat([state, prev_delta], dim=-1)
        
        with torch.no_grad():
            action = self.actor(x)
        
        if add_noise:
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise
            action = torch.clamp(action, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update actor and critic networks."""
        if len(self.buffer) < batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate hedging strategy."""
        batch_size, n_steps, state_dim = features.size()
        device = features.device
        
        deltas = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        with torch.no_grad():
            for k in range(n_steps):
                state = features[:, k, :]
                x = torch.cat([state, prev_delta], dim=-1)
                
                delta = self.actor(x)
                deltas.append(delta.squeeze(-1))
                prev_delta = delta
        
        return torch.stack(deltas, dim=1)


class TD3Hedge(DDPGHedge):
    """
    Twin Delayed DDPG (TD3) for hedging.
    
    Improvements over DDPG:
    - Twin critics to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        buffer_size: int = 100000
    ):
        super().__init__(state_dim, hidden_dim, lr_actor, lr_critic, gamma, tau, noise_std, buffer_size)
        
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0
        
        # Twin critics
        self.critic2 = QNetwork(state_dim + 1, 1, hidden_dim)
        self.critic2_target = QNetwork(state_dim + 1, 1, hidden_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update with TD3 improvements."""
        if len(self.buffer) < batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}
        
        self.update_count += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Target policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            
            # Twin Q-values
            target_q1 = self.critic_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q
        
        # Critic updates
        current_q1 = self.critic(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic2_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # Delayed policy update
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
            self.soft_update(self.critic2_target, self.critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
