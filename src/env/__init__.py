"""Environment utilities and wrappers."""

from .state_representations import (
    FlattenObservation,
    OccupancyGridWrapper,
    GrayscaleObservation,
    create_env_with_representation,
    get_state_dim
)
from .reward_shaping import (
    TTCRewardWrapper,
    SmoothnessRewardWrapper,
    CompositeRewardWrapper,
    create_env_with_reward_shaping
)

__all__ = [
    'FlattenObservation',
    'OccupancyGridWrapper', 
    'GrayscaleObservation',
    'create_env_with_representation',
    'get_state_dim',
    'TTCRewardWrapper',
    'SmoothnessRewardWrapper',
    'CompositeRewardWrapper',
    'create_env_with_reward_shaping'
]