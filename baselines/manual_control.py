"""
Manual control baseline for comparison.

Control the vehicle using keyboard:
- Left/Right arrows: change lane
- Up/Down arrows: accelerate/decelerate

Usage:
    python baselines/manual_control.py
    python baselines/manual_control.py --episodes 10 --save_results
"""

import gymnasium
import highway_env
import numpy as np
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_results', action='store_true')
    return parser.parse_args()


def run_manual_control(args):
    """Run manual control and collect metrics."""
    np.random.seed(args.seed)
    
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'ego_spacing': 1.5,
        'manual_control': True,
    }
    
    env = gymnasium.make("highway-v0", config=env_config, render_mode='human')
    
    print("=" * 50)
    print("MANUAL CONTROL BASELINE")
    print("=" * 50)
    print("Controls: Arrow keys (Left/Right=lane, Up/Down=speed)")
    print(f"Episodes: {args.episodes}")
    print("-" * 50)
    
    # Metrics
    returns = []
    lengths = []
    crashes = []
    
    for episode in range(1, args.episodes + 1):
        env.reset()
        done, truncated = False, False
        episode_return = 0
        episode_steps = 0
        
        while not (done or truncated):
            episode_steps += 1
            # Actions are ignored in manual control mode
            _, reward, done, truncated, _ = env.step(env.action_space.sample())
            env.render()
            episode_return += reward
        
        returns.append(episode_return)
        lengths.append(episode_steps)
        crashes.append(done)
        
        print(f"Episode {episode:3d} | Steps: {episode_steps:4d} | "
              f"Return: {episode_return:8.2f} | Crash: {done}")
    
    env.close()
    
    # Summary
    print("-" * 50)
    print("MANUAL CONTROL SUMMARY")
    print(f"Mean Return:  {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean Length:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Crash Rate:   {np.mean(crashes)*100:.1f}%")
    
    # Save results
    if args.save_results:
        Path("results/tables").mkdir(parents=True, exist_ok=True)
        import csv
        with open('results/tables/manual_control_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'return', 'length', 'crash'])
            for i, (r, l, c) in enumerate(zip(returns, lengths, crashes), 1):
                writer.writerow([i, r, l, c])
        print(f"\nResults saved to results/tables/manual_control_results.csv")
    
    return {'mean_return': np.mean(returns), 'crash_rate': np.mean(crashes)}


if __name__ == "__main__":
    args = parse_args()
    run_manual_control(args)