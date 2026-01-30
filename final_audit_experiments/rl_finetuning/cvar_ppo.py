"""
PART 6: Risk-Aware RL Fine-Tuning

Implements:
- CVaR-PPO: PPO with CVaR-based reward shaping
- TD3 with action penalty for smooth hedging
- Warm-start from trained AttentionLSTM

Reward: r = -CVaR(P&L) - γ|Δδ| - transaction_cost
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class RLConfig:
    """RL training configuration."""
    # PPO
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Learning rates
    lr_policy: float = 3e-5
    lr_value: float = 1e-4
    
    # Training
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Action penalty
    action_penalty: float = 1e-3
    transaction_cost: float = 0.0003  # 3 bps
    
    # CVaR
    cvar_alpha: float = 0.95
    
    # Bounds
    delta_max: float = 1.5


class PolicyNetwork(nn.Module):
    """
    Stochastic policy network for hedging.
    
    Outputs mean and std of Gaussian distribution over delta.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_size: int = 128,
        delta_max: float = 1.5
    ):
        super().__init__()
        
        self.delta_max = delta_max
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std of action distribution."""
        features = self.shared(state)
        mean = self.delta_max * torch.tanh(self.mean(features))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        mean, std = self.forward(state)
        
        if deterministic:
            return mean, None, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Bound action
        action = torch.clamp(action, -self.delta_max, self.delta_max)
        
        return action, log_prob, dist.entropy()


class ValueNetwork(nn.Module):
    """Value network for PPO baseline."""
    
    def __init__(self, state_dim: int, hidden_size: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CVaRPPO:
    """
    CVaR-aware Proximal Policy Optimization.
    
    Modifies reward to penalize tail risk:
    r_t = hedge_pnl - γ|Δδ| - TC - λ * I(in_tail)
    """
    
    def __init__(
        self,
        state_dim: int,
        config: RLConfig,
        device: str = 'cpu'
    ):
        self.config = config
        self.device = device
        
        # Networks
        self.policy = PolicyNetwork(state_dim, delta_max=config.delta_max).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr_policy
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=config.lr_value
        )
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def warm_start_from_model(self, pretrained_model: nn.Module):
        """
        Initialize policy from pretrained supervised model.
        
        This is crucial for stable RL fine-tuning.
        """
        # Copy weights where architecture matches
        pretrained_state = pretrained_model.state_dict()
        policy_state = self.policy.state_dict()
        
        # Try to match layers
        for name, param in pretrained_state.items():
            # Look for matching layer names
            for policy_name, policy_param in policy_state.items():
                if param.shape == policy_param.shape:
                    if 'lstm' in name.lower() or 'fc' in name.lower():
                        print(f"  Warm-starting {policy_name} from {name}")
                        policy_state[policy_name] = param.clone()
                        break
        
        # Load updated state
        self.policy.load_state_dict(policy_state, strict=False)
        print("  Warm-start complete")
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        """Select action using current policy."""
        with torch.no_grad():
            action, log_prob, _ = self.policy.get_action(state, deterministic)
            value = self.value(state)
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        cfg = self.config
        
        rewards = torch.tensor(self.rewards, device=self.device)
        values = torch.cat(self.values)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        # Add next value for bootstrapping
        values = torch.cat([values, next_value])
        
        # GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, next_value: torch.Tensor) -> Dict[str, float]:
        """PPO update step."""
        cfg = self.config
        
        # Compute advantages
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare batches
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, cfg.batch_size):
                end = start + cfg.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get current policy outputs
                mean, std = self.policy(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * batch_advantages.unsqueeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value(batch_states)
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                
                # Update
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), cfg.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        n_updates = cfg.n_epochs * (n_samples // cfg.batch_size + 1)
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }


class HedgingEnvironment:
    """
    Hedging environment for RL training.
    
    State: [S_t/S_0, log(S_t/K), vol_t, tau_t, delta_{t-1}]
    Action: delta_t
    Reward: hedge_pnl - penalties
    """
    
    def __init__(
        self,
        stock_paths: torch.Tensor,
        variance_paths: torch.Tensor,
        payoffs: torch.Tensor,
        K: float,
        T: float,
        config: RLConfig
    ):
        self.stock_paths = stock_paths
        self.variance_paths = variance_paths
        self.payoffs = payoffs
        self.K = K
        self.T = T
        self.config = config
        
        self.n_paths, self.n_steps_plus1 = stock_paths.shape
        self.n_steps = self.n_steps_plus1 - 1
        
        self.current_path = 0
        self.current_step = 0
        self.prev_delta = 0.0
        
        # Track P&L for CVaR computation
        self.episode_pnls = []
    
    def reset(self) -> torch.Tensor:
        """Reset environment to start of new path."""
        self.current_path = (self.current_path + 1) % self.n_paths
        self.current_step = 0
        self.prev_delta = 0.0
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state."""
        t = self.current_step
        path = self.current_path
        
        S = self.stock_paths[path, t].item()
        S0 = self.stock_paths[path, 0].item()
        v = self.variance_paths[path, t].item()
        tau = self.T - t * self.T / self.n_steps
        
        state = torch.tensor([
            S / S0,
            np.log(S / self.K),
            np.sqrt(max(v, 0)),
            tau,
            self.prev_delta
        ], dtype=torch.float32)
        
        return state
    
    def step(self, action: float) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take action and return next state, reward, done, info."""
        t = self.current_step
        path = self.current_path
        cfg = self.config
        
        # Bound action
        delta = np.clip(action, -cfg.delta_max, cfg.delta_max)
        
        # P&L from hedging
        S_t = self.stock_paths[path, t].item()
        S_next = self.stock_paths[path, t + 1].item()
        hedge_pnl = delta * (S_next - S_t)
        
        # Action penalty (smooth deltas)
        action_penalty = cfg.action_penalty * abs(delta - self.prev_delta)
        
        # Transaction cost
        tc = cfg.transaction_cost * abs(delta - self.prev_delta) * S_t
        
        # Reward
        reward = hedge_pnl - action_penalty - tc
        
        # Update state
        self.prev_delta = delta
        self.current_step += 1
        
        # Check if episode done
        done = self.current_step >= self.n_steps
        
        if done:
            # Add final payoff
            final_pnl = -self.payoffs[path].item()
            reward += final_pnl
            self.episode_pnls.append(reward)
            
            # CVaR penalty for tail events
            if len(self.episode_pnls) >= 20:
                pnls = np.array(self.episode_pnls[-100:])
                var = np.percentile(-pnls, cfg.cvar_alpha * 100)
                if -reward >= var:  # This episode is in the tail
                    reward -= 0.5  # Extra penalty for tail events
        
        next_state = self._get_state() if not done else torch.zeros(5)
        
        return next_state, reward, done, {'delta': delta, 'pnl': hedge_pnl}


def train_cvar_ppo(
    agent: CVaRPPO,
    env: HedgingEnvironment,
    n_episodes: int = 1000,
    update_freq: int = 20,
    verbose: bool = True
) -> List[float]:
    """Train CVaR-PPO agent."""
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            state_tensor = state.unsqueeze(0).to(agent.device)
            action, log_prob, value = agent.select_action(state_tensor)
            
            action_np = action.squeeze().cpu().numpy()
            next_state, reward, done, info = env.step(float(action_np))
            
            agent.store_transition(
                state_tensor, action, reward, log_prob, value, done
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update
        if (episode + 1) % update_freq == 0:
            next_value = agent.value(state.unsqueeze(0).to(agent.device))
            losses = agent.update(next_value)
            
            if verbose and (episode + 1) % 100 == 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}: Mean Reward = {mean_reward:.4f}")
    
    return episode_rewards


def run_rl_finetuning(
    pretrained_model: nn.Module = None,
    n_episodes: int = 2000,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run complete RL fine-tuning experiment.
    
    Args:
        pretrained_model: Trained AttentionLSTM to warm-start from
        n_episodes: Number of training episodes
    """
    
    print("=" * 70)
    print("RL FINE-TUNING WITH CVaR-PPO")
    print("=" * 70)
    
    # Generate environment data
    print("\nGenerating environment data...")
    from new_experiments.data.data_generator import DataGenerator, HestonParams
    
    data_gen = DataGenerator(HestonParams())
    train_data, _, _ = data_gen.generate_splits(n_train=10000, n_val=1000, n_test=1000)
    
    # Create config
    config = RLConfig()
    
    # Create environment
    env = HedgingEnvironment(
        stock_paths=train_data.stock_paths,
        variance_paths=train_data.variance_paths,
        payoffs=train_data.payoffs,
        K=100.0,
        T=30/365,
        config=config
    )
    
    # Create agent
    state_dim = 5  # [S_norm, log_money, vol, tau, prev_delta]
    agent = CVaRPPO(state_dim=state_dim, config=config, device=device)
    
    # Warm-start if model provided
    if pretrained_model is not None:
        print("\nWarm-starting from pretrained model...")
        agent.warm_start_from_model(pretrained_model)
    
    # Train
    print(f"\nTraining for {n_episodes} episodes...")
    rewards = train_cvar_ppo(agent, env, n_episodes=n_episodes, verbose=True)
    
    # Results
    results = {
        'final_mean_reward': float(np.mean(rewards[-100:])),
        'reward_history': rewards,
        'config': config.__dict__
    }
    
    print(f"\nFinal mean reward: {results['final_mean_reward']:.4f}")
    
    return results, agent


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results, agent = run_rl_finetuning(device=device)
