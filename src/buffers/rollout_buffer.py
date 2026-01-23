"""Rollout buffer for on-policy algorithms (PPO)."""

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """Stores rollout data for PPO training."""
    
    def __init__(self, buffer_size: int, state_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def push(self, state, action, reward, value, log_prob, done):
        """Store a transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float):
        """Compute GAE advantages and returns."""
        last_gae = 0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
    
    def get(self, batch_size: int, device: torch.device) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Yield mini-batches for training."""
        indices = np.random.permutation(self.ptr)
        
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.FloatTensor(self.states[batch_indices]).to(device),
                torch.LongTensor(self.actions[batch_indices]).to(device),
                torch.FloatTensor(self.log_probs[batch_indices]).to(device),
                torch.FloatTensor(self.advantages[batch_indices]).to(device),
                torch.FloatTensor(self.returns[batch_indices]).to(device),
            )
    
    def reset(self):
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.full = False