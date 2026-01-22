"""
Heuristic baseline for the Autonomous Driving RL project.

This implements a Time-To-Collision (TTC) based policy that:
1. Monitors vehicles ahead for potential collisions
2. Changes lane or slows down when collision risk is high
3. Prefers the rightmost lane (reward bonus)
4. Accelerates when safe

Usage:
    python your_baseline.py
    python your_baseline.py --episodes 10
"""

import gymnasium
import highway_env
import numpy as np
import random
import argparse
from typing import Tuple


def parse_args():
    parser = argparse.ArgumentParser(description='Run heuristic baseline')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to CSV')
    return parser.parse_args()


def set_seed(seed: int):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


class TTCHeuristicPolicy:
    """
    Time-To-Collision based heuristic policy.
    
    State format (5x5 matrix, flattened to 25):
    - Each row: [presence, x, y, vx, vy] for one vehicle
    - Row 0: ego vehicle (absolute reference frame)
    - Rows 1-4: nearby vehicles (relative to ego)
    
    Actions:
    - 0: Lane left
    - 1: Idle
    - 2: Lane right
    - 3: Faster
    - 4: Slower
    """
    
    # Action constants
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4
    
    def __init__(self, 
                 ttc_threshold: float = 3.0,
                 safe_distance: float = 0.15,
                 lane_change_threshold: float = 0.1):
        """
        Initialize the heuristic policy.
        
        Args:
            ttc_threshold: Time-to-collision threshold in seconds
            safe_distance: Minimum safe distance (normalized)
            lane_change_threshold: Y-distance to consider vehicle in same lane
        """
        self.ttc_threshold = ttc_threshold
        self.safe_distance = safe_distance
        self.lane_change_threshold = lane_change_threshold
    
    def compute_ttc(self, x: float, vx: float) -> float:
        """
        Compute Time-To-Collision with a vehicle ahead.
        
        Args:
            x: Relative x position (positive = ahead)
            vx: Relative x velocity (negative = approaching)
        
        Returns:
            TTC in seconds, or inf if no collision risk
        """
        if x <= 0:  # Vehicle behind us
            return float('inf')
        if vx >= 0:  # Vehicle moving away or same speed
            return float('inf')
        
        ttc = -x / vx  # Time until x becomes 0
        return ttc
    
    def get_vehicles_by_lane(self, state: np.ndarray) -> dict:
        """
        Parse state and group vehicles by relative lane position.
        
        Returns:
            Dictionary with 'left', 'same', 'right' keys containing vehicle info
        """
        # Reshape state from (25,) to (5, 5)
        state_matrix = state.reshape(5, 5)
        
        vehicles = {'left': [], 'same': [], 'right': []}
        
        # Skip ego vehicle (row 0), process other vehicles
        for i in range(1, 5):
            presence, x, y, vx, vy = state_matrix[i]
            
            if presence < 0.5:  # Vehicle not present
                continue
            
            vehicle_info = {'x': x, 'y': y, 'vx': vx, 'vy': vy}
            
            # Classify by lane (y position)
            if y < -self.lane_change_threshold:
                vehicles['left'].append(vehicle_info)
            elif y > self.lane_change_threshold:
                vehicles['right'].append(vehicle_info)
            else:
                vehicles['same'].append(vehicle_info)
        
        return vehicles
    
    def is_lane_safe(self, vehicles: list, check_behind: bool = True) -> bool:
        """Check if a lane is safe for lane change."""
        for v in vehicles:
            x, vx = v['x'], v['vx']
            
            # Check vehicles ahead
            if x > 0 and x < self.safe_distance * 2:
                return False
            
            # Check vehicles behind (for lane changes)
            if check_behind and x < 0 and x > -self.safe_distance:
                ttc = self.compute_ttc(-x, -vx)  # Reverse perspective
                if ttc < self.ttc_threshold:
                    return False
        
        return True
    
    def get_min_ttc_ahead(self, vehicles: list) -> Tuple[float, float]:
        """Get minimum TTC and distance to vehicles ahead in current lane."""
        min_ttc = float('inf')
        min_distance = float('inf')
        
        for v in vehicles:
            if v['x'] > 0:  # Vehicle ahead
                ttc = self.compute_ttc(v['x'], v['vx'])
                if ttc < min_ttc:
                    min_ttc = ttc
                if v['x'] < min_distance:
                    min_distance = v['x']
        
        return min_ttc, min_distance
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action based on TTC heuristic.
        
        Decision logic:
        1. If collision imminent (low TTC) → slow down or change lane
        2. If right lane is clear → move right (reward bonus)
        3. If road ahead is clear → accelerate
        4. Default → idle
        """
        vehicles = self.get_vehicles_by_lane(state)
        
        # Get ego vehicle info (row 0 of state)
        state_matrix = state.reshape(5, 5)
        ego_y = state_matrix[0, 2]  # Ego y position (lane indicator)
        
        # Calculate TTC for current lane
        min_ttc, min_distance = self.get_min_ttc_ahead(vehicles['same'])
        
        # 1. DANGER: Collision imminent
        if min_ttc < self.ttc_threshold or min_distance < self.safe_distance:
            # Try to change lane (prefer right for bonus)
            if self.is_lane_safe(vehicles['right']):
                return self.LANE_RIGHT
            elif self.is_lane_safe(vehicles['left']):
                return self.LANE_LEFT
            else:
                # No safe lane change, slow down
                return self.SLOWER
        
        # 2. OPTIMIZATION: Try to move to rightmost lane
        if self.is_lane_safe(vehicles['right']) and len(vehicles['right']) == 0:
            # Right lane is completely clear
            return self.LANE_RIGHT
        
        # 3. SPEED: Road ahead is clear, accelerate
        if min_distance > self.safe_distance * 3:
            return self.FASTER
        
        # 4. DEFAULT: Maintain current state
        return self.IDLE
    
    def __call__(self, state: np.ndarray) -> int:
        """Allow using policy as callable."""
        return self.select_action(state)


def run_baseline(args):
    """Run the heuristic baseline and collect metrics."""
    # Set seed
    set_seed(args.seed)
    
    # Environment configuration
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'ego_spacing': 1.5,
    }
    
    # Create environment
    env_name = "highway-v0"
    render_mode = None if args.no_render else 'human'
    env = gymnasium.make(env_name, config=env_config, render_mode=render_mode)
    
    # Create policy
    policy = TTCHeuristicPolicy(
        ttc_threshold=3.0,
        safe_distance=0.15,
        lane_change_threshold=0.1
    )
    
    print("=" * 50)
    print("TTC HEURISTIC BASELINE")
    print("=" * 50)
    print(f"Environment: {env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"TTC Threshold: {policy.ttc_threshold}s")
    print(f"Safe Distance: {policy.safe_distance}")
    print("-" * 50)
    
    # Metrics
    episode_returns = []
    episode_lengths = []
    crashes = []
    
    # Evaluation loop
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        state = state.reshape(-1)
        done, truncated = False, False
        
        episode_return = 0
        episode_steps = 0
        
        while not (done or truncated):
            episode_steps += 1
            
            # Select action using heuristic
            action = policy.select_action(state)
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.reshape(-1)
            
            if render_mode:
                env.render()
            
            state = next_state
            episode_return += reward
        
        # Store metrics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_steps)
        crashes.append(done)
        
        print(f"Episode {episode:3d} | "
              f"Steps: {episode_steps:4d} | "
              f"Return: {episode_return:8.3f} | "
              f"Crash: {done}")
    
    # Print summary
    print("-" * 50)
    print("BASELINE SUMMARY")
    print("-" * 50)
    print(f"Mean Return:    {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    print(f"Mean Length:    {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Crash Rate:     {np.mean(crashes)*100:.1f}%")
    print(f"Success Rate:   {(1-np.mean(crashes))*100:.1f}%")
    
    env.close()
    
    # Save results if requested
    if args.save_results:
        import pandas as pd
        results_df = pd.DataFrame({
            'episode': range(1, args.episodes + 1),
            'return': episode_returns,
            'length': episode_lengths,
            'crash': crashes
        })
        results_df.to_csv('results/tables/baseline_results.csv', index=False)
        print(f"\nResults saved to results/tables/baseline_results.csv")
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'crash_rate': np.mean(crashes),
    }


if __name__ == "__main__":
    args = parse_args()
    results = run_baseline(args)
