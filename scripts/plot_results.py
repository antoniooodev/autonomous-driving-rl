"""
Generate plots for the report.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --log_dir results/logs --output_dir results/plots
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='results/logs')
    parser.add_argument('--output_dir', type=str, default='results/plots')
    parser.add_argument('--baseline_return', type=float, default=None,
                        help='Baseline return for horizontal line')
    return parser.parse_args()


def load_metrics(log_dir: Path, experiment_name: str) -> pd.DataFrame:
    """Load metrics CSV for an experiment."""
    csv_path = log_dir / experiment_name / "metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def smooth(data, window=10):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def plot_learning_curves(log_dir: Path, output_dir: Path, algorithms: list, baseline_return: float = None):
    """Plot learning curves comparing algorithms."""
    plt.figure(figsize=(10, 6))
    
    colors = {'dqn': '#1f77b4', 'double_dqn': '#ff7f0e', 'dueling_dqn': '#2ca02c', 
              'd3qn': '#d62728', 'ppo': '#9467bd'}
    
    for algo in algorithms:
        df = load_metrics(log_dir, algo)
        if df is None:
            continue
        
        # Get evaluation returns
        if 'eval/return' in df.columns:
            eval_data = df[df['eval/return'].notna()][['step', 'eval/return']]
            if not eval_data.empty:
                plt.plot(eval_data['step'], eval_data['eval/return'], 
                        label=algo.upper().replace('_', ' '), 
                        color=colors.get(algo, None), linewidth=2)
    
    # Baseline horizontal line
    if baseline_return is not None:
        plt.axhline(y=baseline_return, color='gray', linestyle='--', 
                   label='Baseline (TTC Heuristic)', linewidth=2)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Learning Curves: Algorithm Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'learning_curves.png', dpi=150)
    plt.savefig(output_dir / 'learning_curves.pdf')
    plt.close()
    print(f"Saved: learning_curves.png/pdf")


def plot_reward_shaping_comparison(log_dir: Path, output_dir: Path, algorithm: str = 'dqn'):
    """Plot reward shaping experiment results."""
    shapings = ['default', 'ttc', 'smooth', 'composite']
    plt.figure(figsize=(10, 6))
    
    colors = {'default': '#1f77b4', 'ttc': '#ff7f0e', 'smooth': '#2ca02c', 'composite': '#d62728'}
    
    for shaping in shapings:
        experiment_name = f"{algorithm}_{shaping}"
        df = load_metrics(log_dir, experiment_name)
        if df is None:
            continue
        
        if 'eval/return' in df.columns:
            eval_data = df[df['eval/return'].notna()][['step', 'eval/return']]
            if not eval_data.empty:
                plt.plot(eval_data['step'], eval_data['eval/return'],
                        label=shaping.capitalize(), color=colors.get(shaping), linewidth=2)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title(f'Reward Shaping Comparison ({algorithm.upper()})', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'reward_shaping_comparison.png', dpi=150)
    plt.savefig(output_dir / 'reward_shaping_comparison.pdf')
    plt.close()
    print(f"Saved: reward_shaping_comparison.png/pdf")


def plot_env_variations_comparison(log_dir: Path, output_dir: Path, algorithm: str = 'dqn'):
    """Plot environment variations experiment results."""
    variations = ['easy', 'default', 'hard', 'dense']
    plt.figure(figsize=(10, 6))
    
    colors = {'easy': '#2ca02c', 'default': '#1f77b4', 'hard': '#ff7f0e', 'dense': '#d62728'}
    
    for var in variations:
        experiment_name = f"{algorithm}_{var}"
        df = load_metrics(log_dir, experiment_name)
        if df is None:
            continue
        
        if 'eval/return' in df.columns:
            eval_data = df[df['eval/return'].notna()][['step', 'eval/return']]
            if not eval_data.empty:
                plt.plot(eval_data['step'], eval_data['eval/return'],
                        label=var.capitalize(), color=colors.get(var), linewidth=2)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title(f'Environment Variations Comparison ({algorithm.upper()})', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'env_variations_comparison.png', dpi=150)
    plt.savefig(output_dir / 'env_variations_comparison.pdf')
    plt.close()
    print(f"Saved: env_variations_comparison.png/pdf")


def plot_training_metrics(log_dir: Path, output_dir: Path, algorithm: str = 'dqn'):
    """Plot training metrics (loss, epsilon, q_values) for a single algorithm."""
    df = load_metrics(log_dir, algorithm)
    if df is None:
        print(f"No metrics found for {algorithm}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode returns
    if 'train/episode_return' in df.columns:
        returns = df[df['train/episode_return'].notna()]['train/episode_return']
        axes[0, 0].plot(smooth(returns, 50), color='#1f77b4')
        axes[0, 0].set_title('Episode Return (smoothed)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    if 'train/loss' in df.columns:
        loss = df[df['train/loss'].notna()]['train/loss']
        axes[0, 1].plot(smooth(loss, 50), color='#ff7f0e')
        axes[0, 1].set_title('Training Loss (smoothed)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Epsilon
    if 'train/epsilon' in df.columns:
        epsilon = df[df['train/epsilon'].notna()]['train/epsilon']
        axes[1, 0].plot(epsilon.values, color='#2ca02c')
        axes[1, 0].set_title('Exploration (Epsilon)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Q-values
    if 'train/q_mean' in df.columns:
        q_mean = df[df['train/q_mean'].notna()]['train/q_mean']
        axes[1, 1].plot(smooth(q_mean, 50), color='#d62728')
        axes[1, 1].set_title('Mean Q-Value (smoothed)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Q-Value')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Metrics: {algorithm.upper()}', fontsize=14)
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'training_metrics_{algorithm}.png', dpi=150)
    plt.savefig(output_dir / f'training_metrics_{algorithm}.pdf')
    plt.close()
    print(f"Saved: training_metrics_{algorithm}.png/pdf")


def generate_summary_table(log_dir: Path, output_dir: Path, algorithms: list):
    """Generate summary table of final results."""
    results = []
    
    for algo in algorithms:
        df = load_metrics(log_dir, algo)
        if df is None:
            continue
        
        if 'eval/return' in df.columns:
            eval_returns = df[df['eval/return'].notna()]['eval/return']
            if not eval_returns.empty:
                results.append({
                    'Algorithm': algo.upper().replace('_', ' '),
                    'Final Return': f"{eval_returns.iloc[-1]:.2f}",
                    'Max Return': f"{eval_returns.max():.2f}",
                    'Mean Return': f"{eval_returns.mean():.2f}",
                    'Std': f"{eval_returns.std():.2f}"
                })
    
    if results:
        results_df = pd.DataFrame(results)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / 'algorithm_comparison.csv', index=False)
        print(f"\nAlgorithm Comparison:")
        print(results_df.to_string(index=False))
        print(f"\nSaved: algorithm_comparison.csv")
    
    return results


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    
    print("="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    # Find available algorithms
    algorithms = []
    for algo in ['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo']:
        if (log_dir / algo).exists():
            algorithms.append(algo)
    
    print(f"Found algorithms: {algorithms}")
    
    # Generate plots
    if algorithms:
        plot_learning_curves(log_dir, output_dir, algorithms, args.baseline_return)
        generate_summary_table(log_dir, output_dir, algorithms)
        
        for algo in algorithms:
            plot_training_metrics(log_dir, output_dir, algo)
    
    # Reward shaping plots (check if experiments exist)
    for algo in ['dqn', 'd3qn']:
        if (log_dir / f"{algo}_default").exists():
            plot_reward_shaping_comparison(log_dir, output_dir, algo)
            break
    
    # Env variations plots
    for algo in ['dqn', 'd3qn']:
        if (log_dir / f"{algo}_easy").exists():
            plot_env_variations_comparison(log_dir, output_dir, algo)
            break
    
    print("\n" + "="*50)
    print("PLOT GENERATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()