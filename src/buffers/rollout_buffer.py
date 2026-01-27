"""Rollout buffer for on-policy algorithms (PPO)."""

import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self, n_steps: int, num_envs: int, state_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.n_steps = int(n_steps)
        self.num_envs = int(num_envs)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)

        self.states = np.zeros((self.n_steps, self.num_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.num_envs), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)

        self.advantages = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def push(self, states, actions, rewards, values, log_probs, dones):
        """Store a transition batch for one environment step."""
        if self.full:
            return

        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = dones

        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray):
        """Compute GAE advantages and returns."""
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def advantages_flat(self) -> np.ndarray:
        return self.advantages[:self.ptr].reshape(-1)

    def set_advantages_flat(self, adv: np.ndarray):
        self.advantages[:self.ptr] = adv.reshape(self.ptr, self.num_envs)

    def get(self, batch_size: int, device: torch.device) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        total = self.ptr * self.num_envs

        states = self.states[:self.ptr].reshape(total, self.state_dim)
        actions = self.actions[:self.ptr].reshape(total)
        log_probs = self.log_probs[:self.ptr].reshape(total)
        advantages = self.advantages[:self.ptr].reshape(total)
        returns = self.returns[:self.ptr].reshape(total)

        indices = np.random.permutation(total)

        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                torch.FloatTensor(states[batch_indices]).to(device),
                torch.LongTensor(actions[batch_indices]).to(device),
                torch.FloatTensor(log_probs[batch_indices]).to(device),
                torch.FloatTensor(advantages[batch_indices]).to(device),
                torch.FloatTensor(returns[batch_indices]).to(device),
            )

    def reset(self):
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.full = False
