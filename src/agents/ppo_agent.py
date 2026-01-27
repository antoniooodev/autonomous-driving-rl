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
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 256,
        hidden_dims=(64, 64),
    ):
        super().__init__(state_dim, action_dim, device)

        # Hyperparameters
        self.lr = lr
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

        # Vectorization
        self.num_envs = 1
        self._last_values = None
        self._last_log_probs = None

        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, self.num_envs, state_dim, gamma, gae_lambda)

        # For compatibility with single-env code paths
        self._last_value = 0.0
        self._last_log_prob = 0.0

        self.train_steps = 0

    def set_num_envs(self, num_envs: int):
        self.num_envs = max(1, int(num_envs))
        self._last_values = np.zeros(self.num_envs, dtype=np.float32)
        self._last_log_probs = np.zeros(self.num_envs, dtype=np.float32)
        self.buffer = RolloutBuffer(self.n_steps, self.num_envs, self.state_dim, self.gamma, self.gae_lambda)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action from policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, value = self.network(state_t)

            if evaluate:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            self._last_value = float(value.item())
            self._last_log_prob = float(dist.log_prob(action).item())

            return int(action.item())

    def select_action_batch(self, states: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select actions from policy for a batch of states."""
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            dist, values = self.network(states_t)

            if evaluate:
                actions = dist.probs.argmax(dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)

            self._last_values = values.detach().squeeze(-1).cpu().numpy().astype(np.float32, copy=False)
            self._last_log_probs = log_probs.detach().cpu().numpy().astype(np.float32, copy=False)

            return actions.detach().cpu().numpy().astype(np.int64, copy=False)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in rollout buffer."""
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        a = np.asarray([action], dtype=np.int64)
        r = np.asarray([reward], dtype=np.float32)
        v = np.asarray([self._last_value], dtype=np.float32)
        lp = np.asarray([self._last_log_prob], dtype=np.float32)
        d = np.asarray([float(done)], dtype=np.float32)
        self.buffer.push(s, a, r, v, lp, d)

    def store_transition_batch(self, states, actions, rewards, dones):
        """Store a vectorized transition batch in rollout buffer."""
        s = np.asarray(states, dtype=np.float32).reshape(self.num_envs, -1)
        a = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        r = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
        v = np.asarray(self._last_values, dtype=np.float32).reshape(self.num_envs)
        lp = np.asarray(self._last_log_probs, dtype=np.float32).reshape(self.num_envs)
        d = np.asarray(dones, dtype=np.float32).reshape(self.num_envs)
        self.buffer.push(s, a, r, v, lp, d)

    def train_step(self, bootstrap_states: Optional[np.ndarray] = None) -> Optional[dict]:
        """Perform PPO update when buffer is full."""
        if not self.buffer.full:
            return None

        # Compute last values for GAE
        with torch.no_grad():
            if bootstrap_states is None:
                last_values = np.zeros(self.num_envs, dtype=np.float32)
            else:
                bs = np.asarray(bootstrap_states, dtype=np.float32).reshape(self.num_envs, -1)
                bs_t = torch.FloatTensor(bs).to(self.device)
                _, v = self.network(bs_t)
                last_values = v.detach().squeeze(-1).cpu().numpy().astype(np.float32, copy=False)

        self.buffer.compute_returns_and_advantages(last_values)

        # Normalize advantages
        adv = self.buffer.advantages_flat()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.set_advantages_flat(adv)

        # PPO update
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get(self.batch_size, self.device):
                states_b, actions_b, old_log_probs_b, advantages_b, returns_b = batch

                _, new_log_probs, entropy, values = self.network.get_action_and_value(states_b, actions_b)

                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values.squeeze(-1), returns_b)

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.mean().item())
                n_updates += 1

        self.buffer.reset()
        self.train_steps += 1

        if n_updates == 0:
            return None

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
