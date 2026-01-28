"""MLP network architectures."""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """Multi-layer perceptron for Q-value or policy estimation."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """
        __init__.
        
        Args:
            input_dim (int): Parameter.
            output_dim (int): Parameter.
            hidden_dims (List[int]): Parameter.
            activation (str): Parameter.
        """
        super().__init__()
        
        # Select activation function
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU
        }
        act_fn = activations.get(activation, nn.ReLU)
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward.
        
        Args:
            x (torch.Tensor): Parameter.
        
        Returns:
            torch.Tensor: Return value.
        """
        return self.network(x)


# Import numpy for initialization
import numpy as np
