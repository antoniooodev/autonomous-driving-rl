"""PPO Agent implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .base_agent import BaseAgent
from ..networks import ActorCritic
from ..buffers import RolloutBuffer


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto"
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Network
        self.network = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, state_dim, gamma, gae_lambda)
        
        # For storing last value and log_prob
        self._last_value = 0
        self._last_log_prob = 0
        
        self.train_steps = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action from policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, value = self.network(state_t)
            
            if evaluate:
                action = dist.probs.argmax()
            else:
                action = dist.sample()
            
            self._last_value = value.item()
            self._last_log_prob = dist.log_prob(action).item()
            
            return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in rollout buffer."""
        self.buffer.push(state, action, reward, self._last_value, self._last_log_prob, done)
    
    def train_step(self) -> Optional[dict]:
        """Perform PPO update when buffer is full."""
        if not self.buffer.full:
            return None
        
        # Compute last value for GAE
        with torch.no_grad():
            # Use the last stored state to get value
            last_state = torch.FloatTensor(self.buffer.states[self.buffer.ptr - 1]).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_state)
            last_value = last_value.item()
        
        self.buffer.compute_returns_and_advantages(last_value)
        
        # Normalize advantages
        advantages = self.buffer.advantages[:self.buffer.ptr]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages[:self.buffer.ptr] = advantages
        
        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size, self.device):
                states, actions, old_log_probs, advantages_b, returns = batch
                
                # Get current policy outputs
                _, new_log_probs, entropy, values = self.network.get_action_and_value(states, actions)
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        self.buffer.reset()
        self.train_steps += 1
        
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates
        }
    
    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]