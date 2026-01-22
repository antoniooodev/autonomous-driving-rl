"""
Evaluation script for trained RL agents.

Usage:
    python evaluate.py
    python evaluate.py --weights weights/best_model.pth --episodes 10
"""

import gymnasium
import highway_env
import numpy as np
import argparse

from src.utils import set_seed
from src.agents import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--weights', type=str, default='weights/best_model.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_render', action='store_true')
    return parser.parse_args()


def load_agent(algorithm: str, state_dim: int, action_dim: int, weights_path: str):
    """Load trained agent."""
    if algorithm == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    else:
        raise NotImplementedError(f"Algorithm '{algorithm}' not yet implemented")
    
    agent.load(weights_path)
    return agent


def evaluate(args):
    """Main evaluation loop."""
    set_seed(args.seed)
    
    # Environment config
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'ego_spacing': 1.5,
    }
    
    render_mode = None if args.no_render else 'human'
    env = gymnasium.make("highway-v0", config=env_config, render_mode=render_mode)
    
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    print(f"Loading agent from {args.weights}")
    agent = load_agent(args.algorithm, state_dim, action_dim, args.weights)
    
    print(f"Evaluating for {args.episodes} episodes...")
    print("-" * 50)
    
    # Metrics
    returns = []
    lengths = []
    crashes = []
    
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        state = state.reshape(-1)
        done, truncated = False, False
        
        episode_return = 0
        episode_steps = 0
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            state = state.reshape(-1)
            
            if render_mode:
                env.render()
            
            episode_return += reward
            episode_steps += 1
        
        returns.append(episode_return)
        lengths.append(episode_steps)
        crashes.append(done)
        
        print(f"Episode {episode:3d} | Steps: {episode_steps:4d} | Return: {episode_return:8.2f} | Crash: {done}")
    
    # Summary
    print("-" * 50)
    print("EVALUATION SUMMARY")
    print(f"Mean Return:  {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Mean Length:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Crash Rate:   {np.mean(crashes)*100:.1f}%")
    
    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)