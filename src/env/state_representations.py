"""State representation wrappers for highway-env."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenObservation(gym.ObservationWrapper):
    """Flatten observation to 1D vector."""
    
    def __init__(self, env):
        """
        __init__.
        
        Args:
            env (Any): Parameter.
        """
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(np.prod(obs_shape),),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """
        observation.
        
        Args:
            obs (Any): Parameter.
        """
        return obs.reshape(-1).astype(np.float32)


class OccupancyGridWrapper(gym.ObservationWrapper):
    """Convert kinematics to occupancy grid representation."""
    
    def __init__(self, env, grid_size=(32, 32), x_range=(-50, 50), y_range=(-10, 10)):
        """
        __init__.
        
        Args:
            env (Any): Parameter.
            grid_size (Any): Parameter.
            x_range (Any): Parameter.
            y_range (Any): Parameter.
        """
        super().__init__(env)
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size[0], grid_size[1]),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """
        observation.
        
        Args:
            obs (Any): Parameter.
        """
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # obs shape: (vehicles, features) where features = [presence, x, y, vx, vy]
        for vehicle in obs:
            presence, x, y, vx, vy = vehicle
            if presence < 0.5:
                continue
            
            # Denormalize positions (approximate)
            x_pos = x * 100  # Scale factor
            y_pos = y * 10
            
            # Convert to grid coordinates
            grid_x = int((x_pos - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.grid_size[0])
            grid_y = int((y_pos - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.grid_size[1])
            
            # Clamp to grid bounds
            grid_x = np.clip(grid_x, 0, self.grid_size[0] - 1)
            grid_y = np.clip(grid_y, 0, self.grid_size[1] - 1)
            
            grid[grid_x, grid_y] = 1.0
        
        return grid


class GrayscaleObservation(gym.ObservationWrapper):
    """Use grayscale image observation."""
    
    def __init__(self, env):
        """
        __init__.
        
        Args:
            env (Any): Parameter.
        """
        super().__init__(env)
        # Assume env is configured with GrayscaleObservation
        # Shape will be (height, width)
        self.observation_space = env.observation_space
    
    def observation(self, obs):
        # Normalize to [0, 1]
        """
        observation.
        
        Args:
            obs (Any): Parameter.
        """
        return obs.astype(np.float32) / 255.0


def create_env_with_representation(
    env_name: str,
    representation: str = "kinematics",
    flatten: bool = True,
    render_mode: str = None,
    **env_config
):
    """
    Create environment with specified state representation.
    
    Args:
        env_name: Environment name
        representation: One of 'kinematics', 'occupancy_grid', 'grayscale'
        flatten: Whether to flatten kinematics observation
        render_mode: Render mode
        **env_config: Additional environment configuration
    """
    if representation == "kinematics":
        config = {
            'observation': {
                'type': 'Kinematics',
                'vehicles_count': 5,
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
                'normalize': True
            },
            **env_config
        }
        env = gym.make(env_name, config=config, render_mode=render_mode)
        if flatten:
            env = FlattenObservation(env)
    
    elif representation == "occupancy_grid":
        config = {
            'observation': {
                'type': 'OccupancyGrid',
                'features': ['presence', 'on_road'],
                'grid_size': [[-30, 30], [-10, 10]],
                'grid_step': [2, 2],
                'as_image': False
            },
            **env_config
        }
        env = gym.make(env_name, config=config, render_mode=render_mode)
    
    elif representation == "grayscale":
        config = {
            'observation': {
                'type': 'GrayscaleObservation',
                'observation_shape': (64, 64),
                'stack_size': 4,
                'weights': [0.2989, 0.5870, 0.1140]
            },
            **env_config
        }
        env = gym.make(env_name, config=config, render_mode=render_mode)
        env = GrayscaleObservation(env)
    
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    return env


def get_state_dim(representation: str) -> int:
    """Get state dimension for a representation type."""
    dims = {
        "kinematics": 25,  # 5 vehicles * 5 features
        "occupancy_grid": 32 * 32,  # Default grid size
        "grayscale": 64 * 64 * 4  # With frame stacking
    }
    return dims.get(representation, 25)