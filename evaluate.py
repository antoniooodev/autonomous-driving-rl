"""
Main evaluation entry point for the Autonomous Driving RL project.

Usage:
    python evaluate.py
    python evaluate.py --weights weights/best_model.pth
    python evaluate.py --algorithm dqn --episodes 100
"""

import gymnasium
import highway_env
import numpy as np
import torch
import random
import argparse
from pathlib import Path

# Project imports (uncomment as implemented)
# from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'],
                        help='RL algorithm used')
    parser.add_argument('--weights', type=str, default='weights/best_model.pth',
                        help='Path to model weights')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering')
    return parser.parse_args()


def set_seed(seed: int):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_agent(algorithm: str, weights_path: str, state_dim: int, action_dim: int):
    """Load trained agent from weights."""
    # TODO: Implement agent loading
    raise NotImplementedError(f"Agent loading not yet implemented")


def evaluate(args):
    """Main evaluation loop."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Environment configuration (same as training for fair evaluation)
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'ego_spacing': 1.5,
    }
    
    # Create environment
    env_name = "highway-v0"  # Standard version for evaluation
    render_mode = None if args.no_render else 'human'
    env = gymnasium.make(env_name, config=env_config, render_mode=render_mode)
    
    # Get dimensions
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Weights: {args.weights}")
    print(f"Episodes: {args.episodes}")
    print("-" * 50)
    
    # Load agent
    # agent = load_agent(args.algorithm, args.weights, state_dim, action_dim)
    agent = None  # Placeholder
    
    # TODO: Remove this once agent is implemented
    raise NotImplementedError("Agent not yet implemented")
    
    # Evaluation metrics
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
            
            # Select action (greedy, no exploration)
            action = agent.select_action(state, evaluate=True)
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            next_state = next_state.reshape(-1)
            
            if render_mode:
                env.render()
            
            state = next_state
            episode_return += reward
        
        # Log episode results
        episode_returns.append(episode_return)
        episode_lengths.append(episode_steps)
        crashes.append(done)  # done=True means collision
        
        print(f"Episode {episode:3d} | "
              f"Steps: {episode_steps:4d} | "
              f"Return: {episode_return:8.3f} | "
              f"Crash: {done}")
    
    # Print summary statistics
    print("-" * 50)
    print("EVALUATION SUMMARY")
    print("-" * 50)
    print(f"Mean Return:    {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
    print(f"Mean Length:    {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Crash Rate:     {np.mean(crashes)*100:.1f}%")
    print(f"Success Rate:   {(1-np.mean(crashes))*100:.1f}%")
    
    env.close()
    
    return {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'crash_rate': np.mean(crashes),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
    }


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args)
