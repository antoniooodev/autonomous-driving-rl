"""Actor-Critic network for PPO."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple


class ActorCritic(nn.Module):
    """Shared network with separate actor and critic heads."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        
        dist = Categorical(logits=logits)
        return dist, value.squeeze(-1)
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """Get action, log_prob, entropy, and value."""
        dist, value = self.forward(x)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value