"""
Training script for autonomous driving RL agents.

Usage:
    python training.py
    python training.py --algorithm dqn --max_steps 100000
"""

import gymnasium
import highway_env
import argparse
from pathlib import Path
from tqdm import tqdm

from src.utils import set_seed, Logger
from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=100)
    return parser.parse_args()


def create_env(env_name: str, config: dict):
    """Create highway environment."""
    return gymnasium.make(env_name, config=config)


def evaluate_agent(agent, env_name: str, config: dict, n_episodes: int = 5):
    """Evaluate agent over n episodes."""
    env = create_env(env_name, config)
    returns = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        state = state.reshape(-1)
        episode_return = 0
        done, truncated = False, False
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            state = state.reshape(-1)
            episode_return += reward
        
        returns.append(episode_return)
    
    env.close()
    return sum(returns) / len(returns)


def get_agent(algorithm: str, state_dim: int, action_dim: int):
    """Create agent based on algorithm name."""
    if algorithm == 'dqn':
        return DQNAgent(state_dim, action_dim)
    elif algorithm == 'double_dqn':
        return DoubleDQNAgent(state_dim, action_dim)
    elif algorithm == 'dueling_dqn':
        return DuelingDQNAgent(state_dim, action_dim)
    elif algorithm == 'd3qn':
        return D3QNAgent(state_dim, action_dim)
    elif algorithm == 'ppo':
        return PPOAgent(state_dim, action_dim)
    else:
        raise NotImplementedError(f"Algorithm '{algorithm}' not yet implemented")


def train(args):
    """Main training loop."""
    set_seed(args.seed)
    
    # Environment config
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'vehicles_count': 50,
        'duration': 40,
    }
    
    # Create training env
    env = create_env("highway-fast-v0", env_config)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Algorithm: {args.algorithm}, Max steps: {args.max_steps}")
    print("-" * 50)
    
    # Create agent and logger
    agent = get_agent(args.algorithm, state_dim, action_dim)
    logger = Logger("results/logs", experiment_name=args.algorithm)
    
    # Training
    state, _ = env.reset()
    state = state.reshape(-1)
    
    episode = 1
    episode_return = 0
    episode_steps = 0
    
    pbar = tqdm(range(1, args.max_steps + 1), desc="Training", unit="step")
    for step in pbar:
        # Select and execute action
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(-1)
        
        # Store and train
        agent.store_transition(state, action, reward, next_state, done)
        metrics = agent.train_step()
        
        state = next_state
        episode_return += reward
        episode_steps += 1
        
        # Episode end
        if done or truncated:
            pbar.set_postfix({"ep": episode, "ret": f"{episode_return:.1f}"})
            
            log_data = {
                "train/episode_return": episode_return,
                "train/episode_length": episode_steps,
            }
            if hasattr(agent, 'epsilon'):
                log_data["train/epsilon"] = agent.epsilon
            logger.log(log_data, step)
            
            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_return = 0
            episode_steps = 0
        
        # Log training metrics
        if metrics and step % args.log_freq == 0:
            logger.log({f"train/{k}": v for k, v in metrics.items()}, step)
        
        # Evaluation
        if step % args.eval_freq == 0:
            eval_return = evaluate_agent(agent, "highway-fast-v0", env_config)
            tqdm.write(f"[EVAL] Step {step} | Avg Return: {eval_return:.2f}")
            logger.log({"eval/return": eval_return}, step)
        
        # Save checkpoint
        if step % args.save_freq == 0:
            Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
            agent.save(f"results/checkpoints/{args.algorithm}_step{step}.pth")
    
    # Save final model
    Path("weights").mkdir(parents=True, exist_ok=True)
    agent.save("weights/best_model.pth")
    tqdm.write(f"\nTraining complete. Model saved to weights/best_model.pth")
    
    env.close()
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)