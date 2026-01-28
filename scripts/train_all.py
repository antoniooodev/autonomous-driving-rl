"""
Train and compare all algorithms, then select the best model.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --max_steps 50000 --algorithms dqn d3qn ppo
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def parse_args():
    """
    parse_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', nargs='+',
                        default=['dqn', 'double_dqn', 'dueling_dqn', 'd3qn', 'ppo'])
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    """
    main.
    """
    args = parse_args()
    
    print("="*60)
    print("TRAINING ALL ALGORITHMS")
    print("="*60)
    print(f"Algorithms: {args.algorithms}")
    print(f"Max steps: {args.max_steps}")
    print("="*60)
    
    results = {}
    
    for algo in args.algorithms:
        print(f"\n{'='*60}")
        print(f"TRAINING: {algo.upper()}")
        print("="*60)
        
        cmd = [
            sys.executable, "training.py",
            "--algorithm", algo,
            "--max_steps", str(args.max_steps),
            "--seed", str(args.seed)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Move best_model.pth to algorithm-specific name
            best_model = Path("weights/best_model.pth")
            if best_model.exists():
                algo_model = Path(f"weights/{algo}_final.pth")
                shutil.copy(best_model, algo_model)
                print(f"Saved: {algo_model}")
                results[algo] = True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {algo} training failed: {e}")
            results[algo] = False
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for algo, success in results.items():
        status = "✓ Complete" if success else "✗ Failed"
        print(f"{algo:15s}: {status}")
    
    print("\nNext steps:")
    print("1. Run: python scripts/final_evaluation.py")
    print("2. Run: python scripts/plot_results.py")
    print("3. Copy best model to weights/best_model.pth")


if __name__ == "__main__":
    main()