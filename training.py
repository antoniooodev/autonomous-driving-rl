"""
Training script for autonomous driving RL agents.

Usage:
    python training.py
    python training.py --algorithm dqn --max_steps 100000
    python training.py --algorithm d3qn --reward_shaping smooth --max_steps 50000
"""

import os
import argparse
from pathlib import Path

import gymnasium
import highway_env
import numpy as np
import torch
from tqdm import tqdm

from src.utils import set_seed, Logger
from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent
from src.env import SmoothnessRewardWrapper, CompositeRewardWrapper


def get_device():
    """
    get_device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--reward_shaping', type=str, default='none',
                        choices=['none', 'smooth', 'composite'],
                        help='Apply intelligent reward shaping')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, mps, cuda')
    parser.add_argument('--num_envs', type=int, default=0,
                        help='Number of parallel environments when CUDA is available (0 = auto)')
    parser.add_argument('--vector_backend', type=str, default='async',
                        choices=['async', 'sync'],
                        help='Vector environment backend when CUDA is available')
    return parser.parse_args()


def create_env(env_name: str, config: dict, reward_shaping: str = 'none'):
    """
    create_env.
    
    Args:
        env_name (str): Parameter.
        config (dict): Parameter.
        reward_shaping (str): Parameter.
    """
    env = gymnasium.make(env_name, config=config)

    if reward_shaping == 'smooth':
        env = SmoothnessRewardWrapper(env)
        print("Reward shaping: SmoothnessRewardWrapper (intelligent lane change penalty)")
    elif reward_shaping == 'composite':
        env = CompositeRewardWrapper(env)
        print("Reward shaping: CompositeRewardWrapper (TTC + intelligent lane change)")
    elif reward_shaping != 'none':
        print(f"Warning: Unknown reward shaping '{reward_shaping}', using default")

    return env


def evaluate_agent(agent, env_name: str, config: dict, n_episodes: int = 5):
    """
    evaluate_agent.
    
    Args:
        agent (Any): Parameter.
        env_name (str): Parameter.
        config (dict): Parameter.
        n_episodes (int): Parameter.
    """
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


def get_agent(algorithm: str, state_dim: int, action_dim: int, device: str = "auto"):
    """
    get_agent.
    
    Args:
        algorithm (str): Parameter.
        state_dim (int): Parameter.
        action_dim (int): Parameter.
        device (str): Parameter.
    
    Raises:
        NotImplementedError: If an error condition occurs.
    """
    if algorithm == 'dqn':
        return DQNAgent(state_dim, action_dim, device=device)
    if algorithm == 'double_dqn':
        return DoubleDQNAgent(state_dim, action_dim, device=device)
    if algorithm == 'dueling_dqn':
        return DuelingDQNAgent(state_dim, action_dim, device=device)
    if algorithm == 'd3qn':
        return D3QNAgent(state_dim, action_dim, device=device)
    if algorithm == 'ppo':
        return PPOAgent(state_dim, action_dim, device=device)
    raise NotImplementedError(f"Algorithm '{algorithm}' not yet implemented")


def resolve_device(requested: str):
    """
    resolve_device.
    
    Args:
        requested (str): Parameter.
    """
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested
    return get_device()


def configure_cuda():
    """
    configure_cuda.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def make_vector_env(env_name: str, env_config: dict, reward_shaping: str, num_envs: int, base_seed: int, backend: str):
    """
    make_vector_env.
    
    Args:
        env_name (str): Parameter.
        env_config (dict): Parameter.
        reward_shaping (str): Parameter.
        num_envs (int): Parameter.
        base_seed (int): Parameter.
        backend (str): Parameter.
    """
    def make_one(rank: int):
        """
        make_one.
        
        Args:
            rank (int): Parameter.
        """
        def thunk():
            """
            thunk.
            """
            env = create_env(env_name, env_config, reward_shaping)
            env.reset(seed=base_seed + rank)
            return env
        return thunk

    env_fns = [make_one(i) for i in range(num_envs)]

    if backend == "async":
        return gymnasium.vector.AsyncVectorEnv(env_fns)
    return gymnasium.vector.SyncVectorEnv(env_fns)


def select_actions(agent, states, evaluate: bool = False):
    """
    select_actions.
    
    Args:
        agent (Any): Parameter.
        states (Any): Parameter.
        evaluate (bool): Parameter.
    """
    if hasattr(agent, "select_action_batch"):
        return agent.select_action_batch(states, evaluate=evaluate)
    actions = []
    for s in states:
        actions.append(agent.select_action(s, evaluate=evaluate))
    return np.asarray(actions, dtype=np.int64)


def train_single(args, device: str, env_config: dict):
    """
    train_single.
    
    Args:
        args (Any): Parameter.
        device (str): Parameter.
        env_config (dict): Parameter.
    """
    env = create_env("highway-v0", env_config, args.reward_shaping)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Algorithm: {args.algorithm}, Max steps: {args.max_steps}")
    print(f"Device: {device}")
    print("-" * 50)

    agent = get_agent(args.algorithm, state_dim, action_dim, device)
    logger = Logger("results/logs", experiment_name=args.algorithm)

    state, _ = env.reset(seed=args.seed)
    state = state.reshape(-1)

    episode = 1
    episode_return = 0.0
    episode_steps = 0

    pbar = tqdm(range(1, args.max_steps + 1), desc="Training", unit="step")
    for step in pbar:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(-1)

        agent.store_transition(state, action, reward, next_state, done or truncated)
        metrics = agent.train_step(next_state) if args.algorithm == "ppo" else agent.train_step()

        state = next_state
        episode_return += float(reward)
        episode_steps += 1

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
            episode_return = 0.0
            episode_steps = 0

        if metrics and step % args.log_freq == 0:
            logger.log({f"train/{k}": v for k, v in metrics.items()}, step)

        if step % args.eval_freq == 0:
            eval_return = evaluate_agent(agent, "highway-v0", env_config)
            tqdm.write(f"[EVAL] Step {step} | Avg Return: {eval_return:.2f}")
            logger.log({"eval/return": eval_return}, step)

        if step % args.save_freq == 0:
            Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
            agent.save(f"results/checkpoints/{args.algorithm}_step{step}.pth")

    Path("weights").mkdir(parents=True, exist_ok=True)
    agent.save("weights/best_model.pth")
    tqdm.write(f"\nTraining complete. Model saved to weights/best_model.pth")

    env.close()
    logger.close()


def train_vectorized(args, device: str, env_config: dict, num_envs: int):
    """
    train_vectorized.
    
    Args:
        args (Any): Parameter.
        device (str): Parameter.
        env_config (dict): Parameter.
        num_envs (int): Parameter.
    """
    backend = args.vector_backend
    env = make_vector_env("highway-v0", env_config, args.reward_shaping, num_envs, args.seed, backend)

    if hasattr(env, "single_observation_space"):
        obs_space = env.single_observation_space
    else:
        obs_space = env.observation_space

    if hasattr(env, "single_action_space"):
        act_space = env.single_action_space
    else:
        act_space = env.action_space

    state_dim = obs_space.shape[0] * obs_space.shape[1]
    action_dim = act_space.n

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Algorithm: {args.algorithm}, Max steps: {args.max_steps}")
    print(f"Device: {device}")
    print(f"Vectorized envs: {num_envs} ({backend})")
    print("-" * 50)

    agent = get_agent(args.algorithm, state_dim, action_dim, device)
    if args.algorithm == "ppo" and hasattr(agent, "set_num_envs"):
        agent.set_num_envs(num_envs)

    logger = Logger("results/logs", experiment_name=args.algorithm)

    states, _ = env.reset(seed=[args.seed + i for i in range(num_envs)])
    states = np.asarray(states).reshape(num_envs, -1)

    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)

    global_step = 0
    next_log = args.log_freq
    next_eval = args.eval_freq
    next_save = args.save_freq

    pbar = tqdm(total=args.max_steps, desc="Training", unit="step")

    while global_step < args.max_steps:
        remaining = args.max_steps - global_step
        if args.algorithm == "ppo" and remaining < num_envs:
            break

        actions = select_actions(agent, states, evaluate=False)
        next_states, rewards, terminated, truncated, infos = env.step(actions)

        rewards = np.asarray(rewards, dtype=np.float64)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)
        dones = np.logical_or(terminated, truncated)

        next_states = np.asarray(next_states).reshape(num_envs, -1)

        if args.algorithm == "ppo":
            agent.store_transition_batch(states, actions, rewards, dones)
            metrics = agent.train_step(bootstrap_states=next_states)
        else:
            inc = min(num_envs, remaining)

            for i in range(inc):
                agent.store_transition(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]))

            metrics = None
            for _ in range(inc):
                m = agent.train_step()
                if m:
                    metrics = m

        episode_returns += rewards
        episode_lengths += 1

        for i in range(num_envs):
            if dones[i]:
                log_data = {
                    "train/episode_return": float(episode_returns[i]),
                    "train/episode_length": int(episode_lengths[i]),
                }
                if hasattr(agent, 'epsilon'):
                    log_data["train/epsilon"] = agent.epsilon
                logger.log(log_data, global_step + 1)

                episode_returns[i] = 0.0
                episode_lengths[i] = 0

        states = next_states

        inc_steps = num_envs if args.algorithm == "ppo" else min(num_envs, remaining)
        global_step += inc_steps
        pbar.update(inc_steps)

        if metrics and global_step >= next_log:
            logger.log({f"train/{k}": v for k, v in metrics.items()}, global_step)
            next_log += args.log_freq

        while global_step >= next_eval:
            eval_return = evaluate_agent(agent, "highway-v0", env_config)
            tqdm.write(f"[EVAL] Step {next_eval} | Avg Return: {eval_return:.2f}")
            logger.log({"eval/return": eval_return}, next_eval)
            next_eval += args.eval_freq

        while global_step >= next_save:
            Path("results/checkpoints").mkdir(parents=True, exist_ok=True)
            agent.save(f"results/checkpoints/{args.algorithm}_step{next_save}.pth")
            next_save += args.save_freq

    pbar.close()

    Path("weights").mkdir(parents=True, exist_ok=True)
    agent.save("weights/best_model.pth")
    tqdm.write(f"\nTraining complete. Model saved to weights/best_model.pth")

    env.close()
    logger.close()


def train(args):
    """
    train.
    
    Args:
        args (Any): Parameter.
    """
    set_seed(args.seed)

    device = resolve_device(args.device)

    if device == "cuda":
        configure_cuda()
        print("Using CUDA GPU")
    elif device == "mps":
        print("Using MPS (Apple Silicon GPU)")
    else:
        print("Using CPU")

    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'vehicles_count': 50,
        'duration': 40,
    }

    if device == "cuda":
        if args.num_envs <= 0:
            cpu_count = os.cpu_count() or 8
            num_envs = max(2, min(16, cpu_count))
        else:
            num_envs = max(1, int(args.num_envs))
        if num_envs > 1:
            train_vectorized(args, device, env_config, num_envs)
            return

    train_single(args, device, env_config)


if __name__ == "__main__":
    args = parse_args()
    train(args)
