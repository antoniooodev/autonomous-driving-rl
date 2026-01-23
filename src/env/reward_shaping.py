"""Reward shaping wrappers for highway-env."""

import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.RewardWrapper):
    """Base class for reward shaping."""
    
    def reward(self, reward):
        return reward


class TTCRewardWrapper(gym.Wrapper):
    """Add Time-To-Collision penalty to reward."""
    
    def __init__(self, env, ttc_threshold: float = 3.0, ttc_penalty: float = 0.1):
        super().__init__(env)
        self.ttc_threshold = ttc_threshold
        self.ttc_penalty = ttc_penalty
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Compute TTC penalty from observation
        ttc_penalty = self._compute_ttc_penalty(obs)
        shaped_reward = reward - ttc_penalty
        
        return obs, shaped_reward, done, truncated, info
    
    def _compute_ttc_penalty(self, obs):
        """Compute penalty based on minimum TTC."""
        min_ttc = float('inf')
        
        # obs shape: (vehicles, features) = (5, 5)
        for i in range(1, len(obs)):  # Skip ego vehicle
            presence, x, y, vx, vy = obs[i]
            if presence < 0.5:
                continue
            
            # Vehicle ahead in same lane
            if x > 0 and abs(y) < 0.1:
                if vx < 0:  # Approaching
                    ttc = -x / vx
                    min_ttc = min(min_ttc, ttc)
        
        if min_ttc < self.ttc_threshold:
            return self.ttc_penalty * (1 - min_ttc / self.ttc_threshold)
        return 0.0


class SmoothnessRewardWrapper(gym.Wrapper):
    """Penalize frequent lane changes for smoother driving."""
    
    def __init__(self, env, lane_change_penalty: float = 0.1):
        super().__init__(env)
        self.lane_change_penalty = lane_change_penalty
        self.last_action = None
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Penalize lane changes (actions 0 and 2)
        if action in [0, 2]:  # LANE_LEFT or LANE_RIGHT
            reward -= self.lane_change_penalty
        
        # Penalize consecutive different actions (jittery behavior)
        if self.last_action is not None and self.last_action != action:
            if self.last_action in [0, 2] and action in [0, 2]:
                reward -= self.lane_change_penalty * 0.5
        
        self.last_action = action
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)


class CompositeRewardWrapper(gym.Wrapper):
    """Combine multiple reward modifications."""
    
    def __init__(
        self,
        env,
        ttc_penalty: float = 0.1,
        ttc_threshold: float = 3.0,
        lane_change_penalty: float = 0.05,
        speed_bonus_scale: float = 1.0
    ):
        super().__init__(env)
        self.ttc_penalty = ttc_penalty
        self.ttc_threshold = ttc_threshold
        self.lane_change_penalty = lane_change_penalty
        self.speed_bonus_scale = speed_bonus_scale
        self.last_action = None
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Scale base reward
        shaped_reward = reward * self.speed_bonus_scale
        
        # TTC penalty
        shaped_reward -= self._compute_ttc_penalty(obs)
        
        # Lane change penalty
        if action in [0, 2]:
            shaped_reward -= self.lane_change_penalty
        
        self.last_action = action
        return obs, shaped_reward, done, truncated, info
    
    def _compute_ttc_penalty(self, obs):
        min_ttc = float('inf')
        for i in range(1, len(obs)):
            presence, x, y, vx, vy = obs[i]
            if presence < 0.5:
                continue
            if x > 0 and abs(y) < 0.1 and vx < 0:
                ttc = -x / vx
                min_ttc = min(min_ttc, ttc)
        
        if min_ttc < self.ttc_threshold:
            return self.ttc_penalty * (1 - min_ttc / self.ttc_threshold)
        return 0.0
    
    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)


def create_env_with_reward_shaping(
    env_name: str,
    shaping: str = "default",
    render_mode: str = None,
    **env_config
):
    """
    Create environment with specified reward shaping.
    
    Args:
        env_name: Environment name
        shaping: One of 'default', 'ttc', 'smooth', 'composite'
        render_mode: Render mode
        **env_config: Additional environment configuration
    """
    import gymnasium
    import highway_env
    
    base_config = {
        'action': {'type': 'DiscreteMetaAction'},
        **env_config
    }
    
    env = gymnasium.make(env_name, config=base_config, render_mode=render_mode)
    
    if shaping == "default":
        pass  # No modification
    elif shaping == "ttc":
        env = TTCRewardWrapper(env)
    elif shaping == "smooth":
        env = SmoothnessRewardWrapper(env)
    elif shaping == "composite":
        env = CompositeRewardWrapper(env)
    else:
        raise ValueError(f"Unknown reward shaping: {shaping}")
    
    return env