"""
Run final evaluation of all trained models.

Usage:
    python scripts/final_evaluation.py
    python scripts/final_evaluation.py --episodes 100 --algorithms dqn d3qn ppo
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

from src.utils import set_seed
from src.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, PPOAgent


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', nargs='+', 
                        default=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--checkpoints_dir', type=str, default='results/checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/tables')
    return parser.parse_args()


def load_agent(algorithm: str, state_dim: int, action_dim: int, weights_path: str):
    """Load trained agent."""
    agents = {
        'dqn': DQNAgent,
        'double_dqn': DoubleDQNAgent,
        'dueling_dqn': DuelingDQNAgent,
        'd3qn': D3QNAgent,
        'ppo': PPOAgent
    }
    
    if algorithm not in agents:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    agent = agents[algorithm](state_dim, action_dim)
    agent.load(weights_path)
    return agent


def evaluate_agent(agent, env, n_episodes: int, desc: str = "Evaluating"):
    """Evaluate agent for n episodes."""
    returns = []
    lengths = []
    crashes = []
    
    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        state, _ = env.reset()
        state = state.reshape(-1)
        episode_return = 0
        episode_length = 0
        done, truncated = False, False
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            state = state.reshape(-1)
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
        crashes.append(done)  # done=True means crash
    
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'mean_length': np.mean(lengths),
        'crash_rate': np.mean(crashes) * 100,
        'success_rate': (1 - np.mean(crashes)) * 100
    }


def find_weights(algorithm: str, weights_dir: Path, checkpoints_dir: Path):
    """Find best weights for an algorithm."""
    # Try best_model.pth first (if it matches algorithm)
    best_model = weights_dir / "best_model.pth"
    
    # Try algorithm-specific checkpoint
    checkpoints = list(checkpoints_dir.glob(f"{algorithm}_*.pth"))
    if checkpoints:
        # Get latest checkpoint
        checkpoints.sort(key=lambda x: int(x.stem.split('_step')[-1]))
        return checkpoints[-1]
    
    # Fallback to best_model.pth
    if best_model.exists():
        return best_model
    
    return None


def main():
    """
    main.
    """
    args = parse_args()
    set_seed(args.seed)
    
    weights_dir = Path(args.weights_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env_config = {
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'ego_spacing': 1.5,
    }
    env = gymnasium.make("highway-v0", config=env_config)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    
    print("="*60)
    print("FINAL EVALUATION")
    print("="*60)
    print(f"Episodes per algorithm: {args.episodes}")
    print(f"Environment: highway-v0")
    print("-"*60)
    
    results = []
    
    for algo in args.algorithms:
        weights_path = find_weights(algo, weights_dir, checkpoints_dir)
        
        if weights_path is None:
            print(f"[SKIP] {algo}: no weights found")
            continue
        
        print(f"\n[EVAL] {algo.upper()}")
        print(f"       Weights: {weights_path}")
        
        try:
            agent = load_agent(algo, state_dim, action_dim, str(weights_path))
            metrics = evaluate_agent(agent, env, args.episodes, desc=algo)
            
            results.append({
                'Algorithm': algo.upper().replace('_', ' '),
                'Mean Return': f"{metrics['mean_return']:.2f}",
                'Std': f"{metrics['std_return']:.2f}",
                'Max': f"{metrics['max_return']:.2f}",
                'Min': f"{metrics['min_return']:.2f}",
                'Crash Rate %': f"{metrics['crash_rate']:.1f}",
                'Avg Length': f"{metrics['mean_length']:.1f}"
            })
            
            print(f"       Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
            print(f"       Crash Rate: {metrics['crash_rate']:.1f}%")
            
        except Exception as e:
            print(f"[ERROR] {algo}: {e}")
    
    env.close()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / 'final_evaluation.csv', index=False)
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
        print(f"\nSaved to: {output_dir / 'final_evaluation.csv'}")
        
        # Identify best algorithm
        best_idx = df['Mean Return'].apply(lambda x: float(x)).idxmax()
        print(f"\n🏆 Best Algorithm: {df.iloc[best_idx]['Algorithm']}")


if __name__ == "__main__":
    main()