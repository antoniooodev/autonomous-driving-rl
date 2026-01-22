"""D3QN Agent - Double + Dueling + Prioritized Experience Replay."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .base_agent import BaseAgent
from ..networks import DuelingNetwork
from ..buffers import PrioritizedReplayBuffer


class D3QNAgent(BaseAgent):
    """D3QN: combines Double DQN, Dueling architecture, and PER."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        learning_starts: int = 1000,
        target_update_freq: int = 100,
        # PER parameters
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_decay_steps: int = 100000,
        device: str = "auto"
    ):
        super().__init__(state_dim, action_dim, device)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Dueling networks
        self.q_network = DuelingNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DuelingNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(
            buffer_size, state_dim, alpha, beta_start, beta_end, beta_decay_steps
        )
        self.train_steps = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[dict]:
        if len(self.buffer) < self.learning_starts:
            return None
        
        # Sample with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # TD errors for priority update
        td_errors = (q_values - target_q_values).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        # Weighted loss
        loss = (weights * (q_values - target_q_values) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": q_values.mean().item(),
            "beta": self.buffer.beta
        }
    
    def save(self, path: str):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_steps = checkpoint["train_steps"]