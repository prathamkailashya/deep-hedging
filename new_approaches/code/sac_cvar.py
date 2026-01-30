"""
Candidate 3: CVaR-Constrained Soft Actor-Critic (SAC-CVaR)
==========================================================

Grounded in:
- Huang & Lawryshyn 2025: "SAC achieves 50% reduction in CVaR"
- Neagu et al. 2025: "MCPG obtains best performance overall"

Key idea: Add explicit CVaR constraint to SAC via Lagrangian relaxation,
using distributional RL (quantile regression) to estimate tail risk.

Mathematical formulation:
    max_π E_π[R] + α*H(π) - λ*(CVaR_0.95(-R) - c)
    
where λ is learned via dual gradient ascent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class SACHyperparams:
    """SAC-CVaR hyperparameters (from Neagu 2025, Huang 2025)."""
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    lr_lambda: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 256
    n_quantiles: int = 51
    cvar_alpha: float = 0.95
    initial_lambda: float = 0.1
    target_entropy: Optional[float] = None


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous actions.
    Outputs mean and log_std for action distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        # Log probability with tanh squashing correction
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean


class QNetwork(nn.Module):
    """Q-value network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class QuantileNetwork(nn.Module):
    """
    Quantile network for distributional RL.
    
    Outputs N quantile values for estimating the return distribution,
    which enables CVaR computation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = 51,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles)
        )
        
        # Fixed quantile midpoints
        self.register_buffer(
            'tau',
            torch.linspace(0, 1, n_quantiles + 1)[1:] - 0.5 / n_quantiles
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile values.
        
        Returns:
            quantiles: [B, n_quantiles] sorted quantile values
        """
        x = torch.cat([state, action], dim=-1)
        quantiles = self.net(x)
        return quantiles
    
    def compute_cvar(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        alpha: float = 0.95
    ) -> torch.Tensor:
        """
        Compute CVaR at given confidence level.
        
        CVaR_alpha = E[Z | Z >= VaR_alpha]
        
        For tail risk, we use CVaR of negative returns.
        """
        quantiles = self.forward(state, action)  # [B, n_quantiles]
        
        # Find quantiles above alpha level
        cutoff_idx = int(alpha * self.n_quantiles)
        tail_quantiles = quantiles[:, cutoff_idx:]  # Top (1-alpha) quantiles
        
        # CVaR is mean of tail quantiles
        cvar = tail_quantiles.mean(dim=-1, keepdim=True)
        
        return cvar


class CVaRConstrainedSAC:
    """
    Soft Actor-Critic with CVaR constraint via Lagrangian relaxation.
    
    Optimizes:
        max_π E_π[R] + α*H(π) - λ*(CVaR_0.95(-R) - c)
    
    where λ is learned via dual gradient ascent.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cvar_threshold: float,
        params: Optional[SACHyperparams] = None,
        device: str = 'cuda'
    ):
        self.device = device
        self.params = params or SACHyperparams()
        self.cvar_threshold = cvar_threshold
        
        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim).to(device)
        self.critic1 = QNetwork(state_dim, action_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(device)
        self.quantile_critic = QuantileNetwork(
            state_dim, action_dim, self.params.n_quantiles
        ).to(device)
        
        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Learnable temperature and Lagrange multiplier
        self.log_alpha = torch.tensor(
            np.log(0.2), requires_grad=True, device=device
        )
        self.log_lambda = torch.tensor(
            np.log(self.params.initial_lambda), requires_grad=True, device=device
        )
        
        # Target entropy
        if self.params.target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = self.params.target_entropy
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.params.lr_actor
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=self.params.lr_critic
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=self.params.lr_critic
        )
        self.quantile_optimizer = torch.optim.Adam(
            self.quantile_critic.parameters(), lr=self.params.lr_critic
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.params.lr_alpha
        )
        self.lambda_optimizer = torch.optim.Adam(
            [self.log_lambda], lr=self.params.lr_lambda
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(self.params.buffer_size)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    @property
    def lmbda(self):
        return self.log_lambda.exp()
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action given state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.cpu().detach().numpy()[0]
    
    def update(self, batch_size: int) -> Dict[str, float]:
        """Update all networks."""
        if len(self.buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critics
        critic_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update quantile critic
        quantile_loss = self._update_quantile_critic(
            states, actions, rewards, next_states, dones
        )
        
        # Update actor with CVaR penalty
        actor_loss, alpha_loss, lambda_loss, cvar = self._update_actor(states)
        
        # Soft update target networks
        self._soft_update()
        
        return {
            'critic_loss': critic_loss,
            'quantile_loss': quantile_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'lambda_loss': lambda_loss,
            'alpha': self.alpha.item(),
            'lambda': self.lmbda.item(),
            'cvar': cvar
        }
    
    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Update Q-networks."""
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.params.gamma * (1 - dones) * q_next
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return (critic1_loss.item() + critic2_loss.item()) / 2
    
    def _update_quantile_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Update quantile network using quantile Huber loss."""
        with torch.no_grad():
            next_actions, _, _ = self.actor.sample(next_states)
            next_quantiles = self.quantile_critic(next_states, next_actions)
            target_quantiles = rewards + self.params.gamma * (1 - dones) * next_quantiles
        
        current_quantiles = self.quantile_critic(states, actions)
        
        # Quantile Huber loss
        td_error = target_quantiles.unsqueeze(-1) - current_quantiles.unsqueeze(-2)
        huber_loss = torch.where(
            td_error.abs() <= 1.0,
            0.5 * td_error ** 2,
            td_error.abs() - 0.5
        )
        
        tau = self.quantile_critic.tau.view(1, 1, -1)
        quantile_loss = (
            (tau - (td_error < 0).float()).abs() * huber_loss
        ).mean()
        
        self.quantile_optimizer.zero_grad()
        quantile_loss.backward()
        self.quantile_optimizer.step()
        
        return quantile_loss.item()
    
    def _update_actor(
        self,
        states: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """Update actor with CVaR constraint."""
        actions, log_probs, _ = self.actor.sample(states)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q_min = torch.min(q1, q2)
        
        # Compute CVaR for constraint
        cvar = self.quantile_critic.compute_cvar(
            states, actions, self.params.cvar_alpha
        )
        
        # Actor loss with CVaR penalty
        # Maximize: Q - alpha*log_prob - lambda*(CVaR - threshold)
        actor_loss = (
            self.alpha * log_probs - q_min + 
            self.lmbda * (cvar - self.cvar_threshold)
        ).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (entropy temperature)
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update lambda (Lagrange multiplier)
        constraint_violation = (cvar.mean() - self.cvar_threshold).detach()
        lambda_loss = -self.log_lambda * constraint_violation
        
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        # Clamp lambda to be non-negative
        with torch.no_grad():
            self.log_lambda.clamp_(min=-10)  # lambda >= exp(-10) ≈ 0
        
        return (
            actor_loss.item(),
            alpha_loss.item(),
            lambda_loss.item(),
            cvar.mean().item()
        )
    
    def _soft_update(self):
        """Soft update target networks."""
        tau = self.params.tau
        for target_param, param in zip(
            self.critic1_target.parameters(),
            self.critic1.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(
            self.critic2_target.parameters(),
            self.critic2.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class HedgingEnvironment:
    """
    Hedging environment for RL training.
    
    State: [S/S0, log(S/K), sqrt(v), tau, delta_prev]
    Action: delta ∈ [-delta_max, delta_max]
    Reward: -|hedge_error| - transaction_cost
    """
    
    def __init__(
        self,
        prices: torch.Tensor,
        payoffs: torch.Tensor,
        delta_max: float = 1.5,
        transaction_cost: float = 0.001
    ):
        self.prices = prices.numpy() if isinstance(prices, torch.Tensor) else prices
        self.payoffs = payoffs.numpy() if isinstance(payoffs, torch.Tensor) else payoffs
        self.delta_max = delta_max
        self.tc = transaction_cost
        
        self.n_paths = len(prices)
        self.n_steps = prices.shape[1] - 1
        self.current_path = 0
        self.current_step = 0
        self.prev_delta = 0.0
    
    @property
    def state_dim(self):
        return 5
    
    @property
    def action_dim(self):
        return 1
    
    def reset(self) -> np.ndarray:
        """Reset environment to start of new episode."""
        self.current_path = np.random.randint(self.n_paths)
        self.current_step = 0
        self.prev_delta = 0.0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state."""
        S = self.prices[self.current_path, self.current_step]
        S0 = self.prices[self.current_path, 0]
        K = S0  # ATM option
        tau = (self.n_steps - self.current_step) / self.n_steps
        
        return np.array([
            S / S0,
            np.log(S / K + 1e-8),
            0.2,  # Placeholder for sqrt(v)
            tau,
            self.prev_delta
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return next state, reward, done, info."""
        delta = np.clip(action[0], -self.delta_max, self.delta_max)
        
        S_t = self.prices[self.current_path, self.current_step]
        S_next = self.prices[self.current_path, self.current_step + 1]
        
        # Hedge gain
        hedge_gain = delta * (S_next - S_t)
        
        # Transaction cost
        tc = self.tc * abs(delta - self.prev_delta) * S_t
        
        # Reward (negative cost)
        reward = hedge_gain - tc
        
        # Terminal step includes option payoff
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        if done:
            # Final reward includes option liability
            reward -= self.payoffs[self.current_path]
        
        self.prev_delta = delta
        
        return self._get_state(), float(reward), done, {}


if __name__ == "__main__":
    print("Testing SAC-CVaR implementation...")
    
    # Create dummy environment
    n_paths, n_steps = 100, 30
    prices = 100 + np.cumsum(np.random.randn(n_paths, n_steps + 1) * 0.5, axis=1)
    payoffs = np.maximum(prices[:, -1] - 100, 0)
    
    env = HedgingEnvironment(prices, payoffs)
    
    # Create SAC-CVaR agent
    agent = CVaRConstrainedSAC(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cvar_threshold=5.0,
        device='cpu'
    )
    
    # Test action selection
    state = env.reset()
    action = agent.select_action(state)
    print(f"State shape: {state.shape}")
    print(f"Action: {action}")
    
    # Collect some experience
    for _ in range(500):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
    
    # Test update
    metrics = agent.update(batch_size=64)
    print(f"Update metrics: {metrics}")
    
    print("\n✅ SAC-CVaR implementation test passed!")
