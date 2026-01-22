"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract base class that all agents must inherit from."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "auto"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            evaluate: If True, use greedy policy (no exploration)
        
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def train_step(self) -> Optional[dict]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics or None if not enough samples
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent's model parameters."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent's model parameters."""
        pass
