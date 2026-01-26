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
    """
    Intelligent reward shaping that:
    1. Penalizes unnecessary lane changes (no vehicle to overtake)
    2. Rewards staying in right lane when safe
    3. Allows lane changes when needed (slow vehicle ahead)
    
    Observation format (normalized):
    - ego_y: 0.25=left, 0.5=center, 0.75=right (3 lanes)
    - other vehicle y: 0.0=same lane, ±0.25=adjacent lane
    - x: distance ahead (0.0-1.0 range, ~0.02-0.1 is close)
    - vx: relative velocity (negative=approaching)
    """
    
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4
    
    def __init__(
        self, 
        env, 
        unnecessary_lane_change_penalty: float = 0.2,
        right_lane_bonus: float = 0.1,
        zigzag_penalty: float = 0.3,
        safe_distance: float = 0.1,
        slow_vehicle_threshold: float = -0.02
    ):
        super().__init__(env)
        self.unnecessary_lc_penalty = unnecessary_lane_change_penalty
        self.right_lane_bonus = right_lane_bonus
        self.zigzag_penalty = zigzag_penalty
        self.safe_distance = safe_distance
        self.slow_vehicle_threshold = slow_vehicle_threshold
        self.last_action = None
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        shaped_reward = reward
        state_matrix = obs.reshape(5, 5) if obs.ndim == 1 else obs
        
        ego_y = state_matrix[0][2]  # ego lateral position (y)
        
        # 1. ZIGZAG PENALTY (always bad)
        if self.last_action is not None:
            is_zigzag = (
                (action == self.LANE_LEFT and self.last_action == self.LANE_RIGHT) or
                (action == self.LANE_RIGHT and self.last_action == self.LANE_LEFT)
            )
            if is_zigzag:
                shaped_reward -= self.zigzag_penalty
        
        # 2. LANE CHANGE LOGIC
        is_lane_change = action in [self.LANE_LEFT, self.LANE_RIGHT]
        if is_lane_change:
            has_reason = self._has_reason_to_change_lane(state_matrix, action)
            if not has_reason:
                # Unnecessary lane change
                shaped_reward -= self.unnecessary_lc_penalty
        
        # 3. RIGHT LANE BONUS (when safe and not overtaking)
        # ego_y > 0.6 = right lane, ego_y < 0.4 = left lane (3 lanes)
        if not is_lane_change:
            if ego_y > 0.6:  # In right lane
                shaped_reward += self.right_lane_bonus
            elif ego_y < 0.4:  # In left lane
                # Penalize staying left if no one to overtake
                if not self._vehicle_on_right_to_pass(state_matrix):
                    shaped_reward -= self.right_lane_bonus * 0.5
        
        self.last_action = action
        return obs, shaped_reward, done, truncated, info
    
    def _has_reason_to_change_lane(self, state_matrix, action):
        """Check if there's a valid reason for lane change."""
        for i in range(1, 5):
            presence, x, y, vx, vy = state_matrix[i]
            if presence < 0.5:
                continue
            
            # Vehicle ahead in same lane (y close to 0), slower or close
            # Same lane: abs(y) < 0.15 (one lane = 0.25)
            if abs(y) < 0.15 and 0 < x < self.safe_distance * 2:
                if vx < self.slow_vehicle_threshold:  # Slower vehicle
                    return True
                if x < self.safe_distance:  # Very close
                    return True
            
            # Vehicle approaching from behind (need to move)
            if abs(y) < 0.15 and -0.05 < x < 0 and vx > 0.02:
                return True
        
        return False
    
    def _vehicle_on_right_to_pass(self, state_matrix):
        """Check if there's a vehicle on right that we're passing."""
        for i in range(1, 5):
            presence, x, y, vx, vy = state_matrix[i]
            if presence < 0.5:
                continue
            
            # Vehicle on right (y > 0.15), alongside or slightly behind/ahead
            if y > 0.15 and -0.1 < x < 0.15:
                return True
        
        return False
    
    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)


class CompositeRewardWrapper(gym.Wrapper):
    """
    Combined intelligent reward shaping:
    - TTC penalty for dangerous situations
    - Intelligent lane change logic (penalize only unnecessary changes)
    - Right lane preference when safe
    
    Observation format (normalized):
    - ego_y: 0.25=left, 0.5=center, 0.75=right (3 lanes)
    - other vehicle y: 0.0=same lane, ±0.25=adjacent lane
    """
    
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    
    def __init__(
        self,
        env,
        ttc_penalty: float = 0.1,
        ttc_threshold: float = 3.0,
        unnecessary_lc_penalty: float = 0.2,
        right_lane_bonus: float = 0.1,
        zigzag_penalty: float = 0.3,
        speed_bonus_scale: float = 1.0
    ):
        super().__init__(env)
        self.ttc_penalty = ttc_penalty
        self.ttc_threshold = ttc_threshold
        self.unnecessary_lc_penalty = unnecessary_lc_penalty
        self.right_lane_bonus = right_lane_bonus
        self.zigzag_penalty = zigzag_penalty
        self.speed_bonus_scale = speed_bonus_scale
        self.last_action = None
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        shaped_reward = reward * self.speed_bonus_scale
        state_matrix = obs.reshape(5, 5) if obs.ndim == 1 else obs
        
        ego_y = state_matrix[0][2]
        
        # 1. TTC penalty
        shaped_reward -= self._compute_ttc_penalty(state_matrix)
        
        # 2. Zigzag penalty
        if self.last_action is not None:
            is_zigzag = (
                (action == self.LANE_LEFT and self.last_action == self.LANE_RIGHT) or
                (action == self.LANE_RIGHT and self.last_action == self.LANE_LEFT)
            )
            if is_zigzag:
                shaped_reward -= self.zigzag_penalty
        
        # 3. Unnecessary lane change penalty
        is_lane_change = action in [self.LANE_LEFT, self.LANE_RIGHT]
        if is_lane_change:
            if not self._has_reason_to_change_lane(state_matrix):
                shaped_reward -= self.unnecessary_lc_penalty
        
        # 4. Right lane bonus (ego_y > 0.6 = right lane)
        if not is_lane_change and ego_y > 0.6:
            shaped_reward += self.right_lane_bonus
        
        self.last_action = action
        return obs, shaped_reward, done, truncated, info
    
    def _compute_ttc_penalty(self, state_matrix):
        min_ttc = float('inf')
        for i in range(1, 5):
            presence, x, y, vx, vy = state_matrix[i]
            if presence < 0.5:
                continue
            # Same lane (abs(y) < 0.15), vehicle ahead, approaching
            if x > 0 and abs(y) < 0.15 and vx < 0:
                ttc = -x / vx
                min_ttc = min(min_ttc, ttc)
        
        if min_ttc < self.ttc_threshold:
            return self.ttc_penalty * (1 - min_ttc / self.ttc_threshold)
        return 0.0
    
    def _has_reason_to_change_lane(self, state_matrix):
        """Check if there's a valid reason for lane change."""
        for i in range(1, 5):
            presence, x, y, vx, vy = state_matrix[i]
            if presence < 0.5:
                continue
            # Vehicle ahead in same lane, slower or close
            if abs(y) < 0.15 and 0 < x < 0.2:
                if vx < -0.02 or x < 0.1:
                    return True
            # Vehicle approaching from behind
            if abs(y) < 0.15 and -0.05 < x < 0 and vx > 0.02:
                return True
        return False
    
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