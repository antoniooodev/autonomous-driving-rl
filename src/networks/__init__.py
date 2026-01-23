"""Neural network architectures."""

from .mlp import MLP
from .dueling_network import DuelingNetwork
from .actor_critic import ActorCritic

__all__ = ['MLP', 'DuelingNetwork', 'ActorCritic']