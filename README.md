# Autonomous Highway Driving with Deep Reinforcement Learning

<p align="center">
  <img src="https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/highway-env.gif?raw=true" alt="Highway Driving Demo" width="600"/>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#algorithms">Algorithms</a> •
  <a href="#results">Results</a> •
  <a href="#project-structure">Structure</a>
</p>

---

## Overview

This project implements and compares multiple Deep Reinforcement Learning algorithms for autonomous highway driving using the [highway-env](https://github.com/Farama-Foundation/HighwayEnv) simulation environment.

**Objective**: Train an autonomous vehicle to navigate through highway traffic at high speed while avoiding collisions with other vehicles.

### Key Features

- ✅ **5 RL Algorithms**: DQN, Double DQN, Dueling DQN, D3QN, PPO
- ✅ **Baseline Comparison**: TTC Heuristic, Random Policy
- ✅ **Reward Shaping Experiments**: TTC penalty, Smoothness reward, Composite
- ✅ **Action Smoothing**: Post-processing filter for realistic driving behavior
- ✅ **CUDA Vectorized Training**: 3x speedup on NVIDIA GPUs
- ✅ **Apple Silicon Support**: MPS acceleration for Macs (single-env fallback)

---

## Results

All RL agents significantly outperform the heuristic baseline (+15 points, crash rate 89% → 0-2%):

| Algorithm       | Mean Return | Std    | Crash Rate |
| --------------- | ----------- | ------ | ---------- |
| **Dueling DQN** | **28.30**   | 0.96   | **0%**     |
| D3QN            | 28.24       | 0.95   | 0%         |
| DQN             | 28.20       | 0.85   | 0%         |
| Double DQN      | 28.19       | 0.87   | 0%         |
| PPO             | 28.07       | 1.65   | 2%         |
| _TTC Heuristic_ | _13.28_     | _8.87_ | _89%_      |
| _Random_        | _8.57_      | _6.66_ | _96%_      |

_Evaluation: 100 episodes, seed e-1 for episode e, ActionSmoother enabled_

<p align="center">
  <img src="https://i.ibb.co/tphJ2KCT/learning-curves.png" alt="Learning Curves" width="700"/>
</p>

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/antoniooodev/autonomous-driving-rl.git
cd autonomous-driving-rl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dependencies for CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Dependencies

```
torch>=2.0.0
gymnasium>=0.29.0
highway-env>=1.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.13.0
```

---

## Quick Start

### Evaluate Pre-trained Agent

```bash
# Run evaluation with the best model (Dueling DQN)
python evaluate.py

# Evaluate specific algorithm
python evaluate.py --algorithm d3qn --weights weights/final/d3qn_step50000.pth --episodes 10

# Evaluate without rendering (faster, 100 episodes)
python evaluate.py --no_render --episodes 100
```

### Train from Scratch

```bash
# Train DQN (default)
python training.py --algorithm dqn --max_steps 50000

# Train with CUDA vectorized environments (NVIDIA GPU, ~3x faster)
python training.py --algorithm dqn --max_steps 50000 --num_envs 8

# Train with reward shaping
python training.py --algorithm d3qn --reward_shaping smooth --max_steps 50000

# Train all algorithms
python training.py --algorithm dqn --max_steps 50000
python training.py --algorithm double_dqn --max_steps 50000
python training.py --algorithm dueling_dqn --max_steps 50000
python training.py --algorithm d3qn --max_steps 50000
python training.py --algorithm ppo --max_steps 50000
```

### Run Baseline

```bash
# TTC Heuristic and Random baseline
python scripts/evaluate_baselines.py

# Generate plots from baseline results
python scripts/plot_results.py --baseline_return xx.xx
```

### Run Manual Control

```bash
# Manual control baseline
python baseline/manual_control.py
```

---

## Algorithms

### Value-Based Methods

| Algorithm       | Description                                                          |
| --------------- | -------------------------------------------------------------------- |
| **DQN**         | Deep Q-Network with experience replay and target network             |
| **Double DQN**  | Reduces overestimation by decoupling action selection and evaluation |
| **Dueling DQN** | Separates state-value and advantage streams: Q(s,a) = V(s) + A(s,a)  |
| **D3QN**        | Combines Double DQN + Dueling + Prioritized Experience Replay        |

### Policy Gradient Methods

| Algorithm | Description                                                   |
| --------- | ------------------------------------------------------------- |
| **PPO**   | Proximal Policy Optimization with clipped surrogate objective |

### Action Smoothing

To prevent jittery lane-change behavior, an `ActionSmoother` is implemented that:

- Enforces minimum steps (default: 3) between lane changes
- Allows emergency maneuvers when collision is imminent (x < 0.1 normalized)
- Prevents zigzag patterns (LEFT→RIGHT→LEFT)

```bash
# Disable smoothing to see raw policy behavior
python evaluate.py --no_smooth
```

---

## Project Structure

```
autonomous-driving-rl/
├── training.py              # Main training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
├── baseline.py
│
├── src/
│   ├── agents/              # RL agent implementations
│   │   ├── dqn_agent.py
│   │   ├── double_dqn_agent.py
│   │   ├── dueling_dqn_agent.py
│   │   ├── d3qn_agent.py
│   │   └── ppo_agent.py
│   │
│   ├── networks/            # Neural network architectures
│   │   ├── mlp.py
│   │   ├── dueling_network.py
│   │   └── actor_critic.py
│   │
│   ├── buffers/             # Replay buffers
│   │   ├── replay_buffer.py
│   │   ├── prioritized_replay.py
│   │   └── rollout_buffer.py
│   │
│   ├── env/                 # Environment wrappers
│   │   └── reward_shaping.py
│   │
│   └── utils/               # Utilities
│       └── logger.py
│
├── configs/                 # Configuration files
│   ├── default.yaml
│   ├── algorithms/
│   └── experiments/
│
├── weights/                 # Trained models
│   ├── best_model.pth       # Best model (Dueling DQN)
│   └── final/               # All final models
│
├── results/
│   ├── logs/                # Training logs (TensorBoard)
│   ├── plots/               # Generated figures
│   └── tables/              # CSV results
│
├── report/                  # Pdf report
│   └── antonio_tangaro_autonomus_driving.pdf
│
│
│
└── scripts/                 # Utility scripts
    ├── evaluate_baselines.py
    ├── plot_results.py
    └── final_evaluation.py
```

---

## Experiments

### Reward Shaping

Different reward modifications were tested:

| Shaping   | Description                       | Result              |
| --------- | --------------------------------- | ------------------- |
| Default   | Original highway-env reward       | Baseline            |
| TTC       | Penalize low time-to-collision    | Similar performance |
| Smooth    | Penalize unnecessary lane changes | Lower performance   |
| Composite | TTC + Smooth combined             | Similar to default  |

<p align="center">
  <img src="https://i.ibb.co/VW1sf8FX/reward-shaping-comparison.png" alt="Reward Shaping" width="700"/>
</p>

**Finding**: Naive reward shaping can hurt performance. The default reward function is already well-tuned. Post-processing (ActionSmoother) is more effective than reward modification.

### Environment Variations

| Environment | Lanes | Vehicles | Mean Return |
| ----------- | ----- | -------- | ----------- |
| Easy        | 4     | 20       | 28.34       |
| Default     | 3     | 50       | 27.95       |
| Dense       | 3     | 100      | 27.73       |
| Hard        | 2     | 80       | 24.71       |

---

## Configuration

### Training Arguments

```bash
python training.py --help

Options:
  --algorithm       Algorithm to train (dqn, double_dqn, dueling_dqn, d3qn, ppo)
  --max_steps       Maximum training steps (default: 100000)
  --seed            Random seed (default: 0)
  --eval_freq       Evaluation frequency (default: 5000)
  --save_freq       Checkpoint save frequency (default: 10000)
  --reward_shaping  Reward shaping type (none, smooth, composite)
  --num_envs        Number of parallel environments for CUDA (default: 1)
  --vector_backend  Vectorization backend: async or sync (default: async)
```

### Evaluation Arguments

```bash
python evaluate.py --help

Options:
  --algorithm       Algorithm to evaluate (default: dueling_dqn)
  --weights         Path to model weights (default: weights/best_model.pth)
  --episodes        Number of evaluation episodes (default: 10)
  --seed            Random seed (default: 0)
  --no_render       Disable rendering
  --no_smooth       Disable action smoothing
  --smooth_window   Minimum steps between lane changes (default: 3)
```

---

## Reproducibility

All experiments use **seed=0** for reproducibility. Evaluation episodes use deterministic seeding: episode _e_ uses seed _e-1_, ensuring identical traffic configurations across runs.

```bash
# Reproduce final results
python scripts/evaluate_baselines.py
python evaluate.py --algorithm dueling_dqn --episodes 100 --no_render
```

---

## Training Performance

| Device                          | Training Time (50k steps) |
| ------------------------------- | ------------------------- |
| NVIDIA GPU (CUDA, vectorized)   | ~30 min                   |
| Apple Silicon (MPS, single-env) | ~1h 30min                 |
| CPU                             | ~3h+                      |

Use `--num_envs 8` on CUDA for maximum speedup.

---

## Course Project

This project was developed for the **Reinforcement Learning** course (2025/2026) at the University of Padova.

**Professor**: Gian Antonio Susto

**Student**: Antonio Tangaro (ID: 2163822)
