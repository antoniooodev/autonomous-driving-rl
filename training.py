"""
Main training entry point for the Autonomous Driving RL project.

Usage:
    python training.py
    python training.py --algorithm dqn
    python training.py --algorithm ppo --config configs/algorithms/ppo.yaml
"""

import gymnasium
import highway_env
import numpy as np
import torch
import random
import argparse
from pathlib import Path

# Project imports (uncomment as implemented)
# from src.utils.config import load_config
# from src.utils.logger import Logger
# from src.utils.seed import set_seed
# from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for autonomous driving')
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'],
                        help='RL algorithm to use')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    return parser.parse_args()


def set_seed(seed: int):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_env(env_name: str, config: dict, render: bool = False):
    """Create and configure the highway environment."""
    render_mode = 'human' if render else None
    env = gymnasium.make(env_name, config=config, render_mode=render_mode)
    return env


def get_agent(algorithm: str, state_dim: int, action_dim: int, config: dict):
    """Factory function to create the appropriate agent."""
    # TODO: Implement agent creation
    raise NotImplementedError(f"Agent '{algorithm}' not yet implemented")


def train(args):
    """Main training loop."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Environment configuration
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'vehicles_count': 50,
        'duration': 40,
        'ego_spacing': 1.5,
    }
    
    # Create environment
    env_name = "highway-fast-v0"  # Fast version for training
    env = create_env(env_name, env_config, render=args.render)
    
    # Get dimensions
    state_dim = np.prod(env.observation_space.shape)  # Flattened: 5*5 = 25
    action_dim = env.action_space.n  # 5 actions
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Max steps: {args.max_steps}")
    print("-" * 50)
    
    # Initialize agent
    # agent = get_agent(args.algorithm, state_dim, action_dim, config={})
    agent = None  # Placeholder
    
    # TODO: Implement training loop
    raise NotImplementedError("Training loop not yet implemented")
    
    # Training loop structure:
    # state, _ = env.reset()
    # state = state.reshape(-1)
    # 
    # for step in range(args.max_steps):
    #     # Select action
    #     action = agent.select_action(state)
    #     
    #     # Environment step
    #     next_state, reward, done, truncated, info = env.step(action)
    #     next_state = next_state.reshape(-1)
    #     
    #     # Store transition and train
    #     agent.store_transition(state, action, reward, next_state, done)
    #     agent.train_step()
    #     
    #     state = next_state
    #     
    #     if done or truncated:
    #         state, _ = env.reset()
    #         state = state.reshape(-1)
    #
    # # Save final model
    # agent.save('weights/best_model.pth')
    
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
