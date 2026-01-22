"""Neural network architectures."""

from .mlp import MLP
from .dueling_network import DuelingNetwork

__all__ = ['MLP', 'DuelingNetwork']