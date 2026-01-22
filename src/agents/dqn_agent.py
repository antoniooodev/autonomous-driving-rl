"""DQN Agent implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .base_agent import BaseAgent
from ..networks import MLP
from ..buffers import ReplayBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network agent."""
    
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
        device: str = "auto"
    ):
        super().__init__(state_dim, action_dim, device)
        
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Networks
        self.q_network = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = MLP(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        
        # Training step counter
        self.train_steps = 0
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[dict]:
        """Perform one training step."""
        if len(self.buffer) < self.learning_starts:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        return {"loss": loss.item(), "epsilon": self.epsilon, "q_mean": q_values.mean().item()}
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps
        }, path)
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_steps = checkpoint["train_steps"]