"""
Evaluate baseline policies and save results.

Usage:
    python scripts/evaluate_baselines.py
    python scripts/evaluate_baselines.py --episodes 100
"""

import argparse
import gymnasium
import highway_env
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TTCHeuristicPolicy:
    """Time-To-Collision based heuristic policy."""
    
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4
    
    def __init__(self, ttc_threshold=3.0, safe_distance=0.15, lane_change_threshold=0.1):
        """
        __init__.
        
        Args:
            ttc_threshold (Any): Parameter.
            safe_distance (Any): Parameter.
            lane_change_threshold (Any): Parameter.
        """
        self.ttc_threshold = ttc_threshold
        self.safe_distance = safe_distance
        self.lane_change_threshold = lane_change_threshold
    
    def select_action(self, state):
        """
        select_action.
        
        Args:
            state (Any): Parameter.
        """
        state_matrix = state.reshape(5, 5)
        vehicles = self._get_vehicles_by_lane(state_matrix)
        min_ttc, min_distance = self._get_min_ttc_ahead(vehicles['same'])
        
        if min_ttc < self.ttc_threshold or min_distance < self.safe_distance:
            if self._is_lane_safe(vehicles['right']):
                return self.LANE_RIGHT
            elif self._is_lane_safe(vehicles['left']):
                return self.LANE_LEFT
            else:
                return self.SLOWER
        
        if self._is_lane_safe(vehicles['right']) and len(vehicles['right']) == 0:
            return self.LANE_RIGHT
        
        if min_distance > self.safe_distance * 3:
            return self.FASTER
        
        return self.IDLE
    
    def _get_vehicles_by_lane(self, state_matrix):
        """
        _get_vehicles_by_lane.
        
        Args:
            state_matrix (Any): Parameter.
        """
        vehicles = {'left': [], 'same': [], 'right': []}
        for i in range(1, 5):
            presence, x, y, vx, vy = state_matrix[i]
            if presence < 0.5:
                continue
            v = {'x': x, 'y': y, 'vx': vx, 'vy': vy}
            if y < -self.lane_change_threshold:
                vehicles['left'].append(v)
            elif y > self.lane_change_threshold:
                vehicles['right'].append(v)
            else:
                vehicles['same'].append(v)
        return vehicles
    
    def _is_lane_safe(self, vehicles):
        """
        _is_lane_safe.
        
        Args:
            vehicles (Any): Parameter.
        """
        for v in vehicles:
            if v['x'] > 0 and v['x'] < self.safe_distance * 2:
                return False
            if v['x'] < 0 and v['x'] > -self.safe_distance:
                return False
        return True
    
    def _get_min_ttc_ahead(self, vehicles):
        """
        _get_min_ttc_ahead.
        
        Args:
            vehicles (Any): Parameter.
        """
        min_ttc = float('inf')
        min_distance = float('inf')
        for v in vehicles:
            if v['x'] > 0:
                if v['vx'] < 0:
                    ttc = -v['x'] / v['vx']
                    min_ttc = min(min_ttc, ttc)
                min_distance = min(min_distance, v['x'])
        return min_ttc, min_distance


class RandomPolicy:
    """Random action policy."""
    def __init__(self, action_dim=5):
        """
        __init__.
        
        Args:
            action_dim (Any): Parameter.
        """
        self.action_dim = action_dim
    
    def select_action(self, state):
        """
        select_action.
        
        Args:
            state (Any): Parameter.
        """
        return np.random.randint(self.action_dim)


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results/tables')
    return parser.parse_args()


def evaluate_policy(policy, env, n_episodes, desc="Evaluating"):
    """Evaluate a policy for n episodes."""
    returns = []
    lengths = []
    crashes = []
    
    for episode in tqdm(range(n_episodes), desc=desc):
        state, _ = env.reset(seed=episode)
        state = state.reshape(-1)
        episode_return = 0
        episode_length = 0
        done, truncated = False, False
        
        while not (done or truncated):
            action = policy.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            state = state.reshape(-1)
            episode_return += reward
            episode_length += 1
        
        crashed = info.get("crashed", done and not truncated)
        
        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(crashed)

    
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'mean_length': np.mean(lengths),
        'crash_rate': np.mean(crashes) * 100
    }


def main():
    """
    main.
    """
    args = parse_args()
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'ego_spacing': 1.5,
    }
    env = gymnasium.make("highway-v0", config=env_config)
    
    print("="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print("-"*60)
    
    results = []
    
    # TTC Heuristic
    print("\nEvaluating TTC Heuristic...")
    ttc_policy = TTCHeuristicPolicy()
    ttc_metrics = evaluate_policy(ttc_policy, env, args.episodes, "TTC Heuristic")
    results.append({
        'Policy': 'TTC Heuristic',
        'Mean Return': f"{ttc_metrics['mean_return']:.2f}",
        'Std': f"{ttc_metrics['std_return']:.2f}",
        'Max': f"{ttc_metrics['max_return']:.2f}",
        'Crash Rate %': f"{ttc_metrics['crash_rate']:.1f}",
        'Avg Length': f"{ttc_metrics['mean_length']:.1f}"
    })
    
    # Random policy
    print("\nEvaluating Random Policy...")
    random_policy = RandomPolicy()
    random_metrics = evaluate_policy(random_policy, env, args.episodes, "Random")
    results.append({
        'Policy': 'Random',
        'Mean Return': f"{random_metrics['mean_return']:.2f}",
        'Std': f"{random_metrics['std_return']:.2f}",
        'Max': f"{random_metrics['max_return']:.2f}",
        'Crash Rate %': f"{random_metrics['crash_rate']:.1f}",
        'Avg Length': f"{random_metrics['mean_length']:.1f}"
    })
    
    env.close()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'baseline_evaluation.csv', index=False)
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_dir / 'baseline_evaluation.csv'}")
    
    # Print value for plot_results.py
    print(f"\nFor learning curves plot, use:")
    print(f"  --baseline_return {ttc_metrics['mean_return']:.2f}")


if __name__ == "__main__":
    main()