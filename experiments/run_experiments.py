"""
Run experiments for bonus comparisons.

Usage:
    python experiments/run_experiments.py --experiment reward_shaping --algorithm dqn
    python experiments/run_experiments.py --experiment env_variations --algorithm d3qn
"""

import argparse
import gymnasium
import highway_env
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, Logger, load_config
from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent
from src.env import create_env_with_reward_shaping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['reward_shaping', 'env_variations'])
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=5000)
    return parser.parse_args()


def get_agent(algorithm: str, state_dim: int, action_dim: int):
    agents = {
        'dqn': DQNAgent,
        'double_dqn': DoubleDQNAgent,
        'dueling_dqn': DuelingDQNAgent,
        'd3qn': D3QNAgent,
        'ppo': PPOAgent
    }
    return agents[algorithm](state_dim, action_dim)


def evaluate(agent, env, n_episodes=5):
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
    return np.mean(returns)


def run_reward_shaping_experiment(args):
    """Run reward shaping comparison."""
    shapings = ['default', 'ttc', 'smooth', 'composite']
    results = {}
    
    for shaping in shapings:
        print(f"\n{'='*50}")
        print(f"Running: {shaping} reward shaping with {args.algorithm}")
        print('='*50)
        
        set_seed(args.seed)
        
        env = create_env_with_reward_shaping(
            "highway-fast-v0",
            shaping=shaping,
            lanes_count=3,
            vehicles_count=50,
            duration=40
        )
        
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.n
        
        agent = get_agent(args.algorithm, state_dim, action_dim)
        logger = Logger("results/logs", f"{args.algorithm}_{shaping}")
        
        state, _ = env.reset()
        state = state.reshape(-1)
        episode = 1
        episode_return = 0
        eval_returns = []
        
        pbar = tqdm(range(1, args.max_steps + 1), desc=f"{shaping}", unit="step")
        for step in pbar:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state.reshape(-1)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_return += reward
            
            if done or truncated:
                pbar.set_postfix({"ep": episode, "ret": f"{episode_return:.1f}"})
                state, _ = env.reset()
                state = state.reshape(-1)
                episode += 1
                episode_return = 0
            
            if step % args.eval_freq == 0:
                eval_return = evaluate(agent, env)
                eval_returns.append(eval_return)
                tqdm.write(f"[{shaping}] Step {step} | Eval Return: {eval_return:.2f}")
                logger.log({"eval/return": eval_return}, step)
        
        env.close()
        logger.close()
        results[shaping] = eval_returns
    
    # Print summary
    print(f"\n{'='*50}")
    print("REWARD SHAPING SUMMARY")
    print('='*50)
    for shaping, returns in results.items():
        print(f"{shaping:15s}: Final={returns[-1]:.2f}, Max={max(returns):.2f}")


def run_env_variations_experiment(args):
    """Run environment variations comparison."""
    variations = {
        'easy': {'lanes_count': 4, 'vehicles_count': 20},
        'default': {'lanes_count': 3, 'vehicles_count': 50},
        'hard': {'lanes_count': 2, 'vehicles_count': 80},
        'dense': {'lanes_count': 3, 'vehicles_count': 100}
    }
    results = {}
    
    for name, config in variations.items():
        print(f"\n{'='*50}")
        print(f"Running: {name} environment with {args.algorithm}")
        print('='*50)
        
        set_seed(args.seed)
        
        env_config = {
            'action': {'type': 'DiscreteMetaAction'},
            'duration': 40,
            **config
        }
        env = gymnasium.make("highway-fast-v0", config=env_config)
        
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.n
        
        agent = get_agent(args.algorithm, state_dim, action_dim)
        logger = Logger("results/logs", f"{args.algorithm}_{name}")
        
        state, _ = env.reset()
        state = state.reshape(-1)
        episode = 1
        episode_return = 0
        eval_returns = []
        
        pbar = tqdm(range(1, args.max_steps + 1), desc=f"{name}", unit="step")
        for step in pbar:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = next_state.reshape(-1)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_return += reward
            
            if done or truncated:
                pbar.set_postfix({"ep": episode, "ret": f"{episode_return:.1f}"})
                state, _ = env.reset()
                state = state.reshape(-1)
                episode += 1
                episode_return = 0
            
            if step % args.eval_freq == 0:
                eval_return = evaluate(agent, env)
                eval_returns.append(eval_return)
                tqdm.write(f"[{name}] Step {step} | Eval Return: {eval_return:.2f}")
                logger.log({"eval/return": eval_return}, step)
        
        env.close()
        logger.close()
        results[name] = eval_returns
    
    # Print summary
    print(f"\n{'='*50}")
    print("ENVIRONMENT VARIATIONS SUMMARY")
    print('='*50)
    for name, returns in results.items():
        print(f"{name:15s}: Final={returns[-1]:.2f}, Max={max(returns):.2f}")


def main():
    args = parse_args()
    
    if args.experiment == 'reward_shaping':
        run_reward_shaping_experiment(args)
    elif args.experiment == 'env_variations':
        run_env_variations_experiment(args)


if __name__ == "__main__":
    main()