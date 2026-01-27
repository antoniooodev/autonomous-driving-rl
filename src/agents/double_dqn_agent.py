"""Double DQN Agent - reduces overestimation bias."""

import numpy as np
import torch
import torch.nn as nn

from .dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN: uses online network to select actions,
    target network to evaluate them.
    """

    def select_action_batch(self, states: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select actions for a batch of states using epsilon-greedy policy."""
        return super().select_action_batch(states, evaluate=evaluate)
    
    def train_step(self):
        """Training step with Double DQN target computation."""
        if len(self.buffer) < self.learning_starts:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        
        return {"loss": loss.item(), "epsilon": self.epsilon}