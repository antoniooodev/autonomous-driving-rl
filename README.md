# 🚗 Autonomous Highway Driving with Deep Reinforcement Learning

<p align="center">
  <img src="results/plots/highway_demo.gif" alt="Highway Driving Demo" width="600"/>
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
- ✅ **Baseline Comparison**: TTC Heuristic, Random Policy, Manual Control
- ✅ **Reward Shaping Experiments**: TTC penalty, Smoothness reward, Composite
- ✅ **Action Smoothing**: Post-processing filter for realistic driving behavior
- ✅ **Apple Silicon Support**: MPS acceleration for M1/M2 Macs

---

## 🎯 Results

All RL agents significantly outperform the heuristic baseline:

| Algorithm       | Mean Return | Std    | Crash Rate |
| --------------- | ----------- | ------ | ---------- |
| **Dueling DQN** | **28.00**   | 0.94   | **0%**     |
| D3QN            | 27.96       | 0.92   | 0%         |
| Double DQN      | 27.89       | 1.59   | 1%         |
| DQN             | 27.64       | 0.85   | 0%         |
| PPO             | 27.46       | 3.68   | 4%         |
| _TTC Heuristic_ | _14.76_     | _9.28_ | _85%_      |
| _Random_        | _7.92_      | _5.71_ | _99%_      |

<p align="center">
  <img src="results/plots/learning_curves.png" alt="Learning Curves" width="700"/>
</p>

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/autonomous-driving-rl.git
cd autonomous-driving-rl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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

## ⚡ Quick Start

### Evaluate Pre-trained Agent

```bash
# Run evaluation with the best model (Dueling DQN)
python evaluate.py

# Evaluate specific algorithm
python evaluate.py --algorithm d3qn --weights weights/final/d3qn_step50000.pth --episodes 10

# Evaluate without rendering (faster)
python evaluate.py --no_render --episodes 100
```

### Train from Scratch

```bash
# Train DQN (default)
python training.py --algorithm dqn --max_steps 50000

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
# TTC Heuristic baseline
python scripts/evaluate_baselines.py
```

---

## 🧠 Algorithms

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

To prevent jittery lane-change behavior, I implement an `ActionSmoother` that:

- Enforces minimum steps between lane changes
- Allows emergency maneuvers when collision is imminent
- Prevents zigzag patterns (LEFT→RIGHT→LEFT)

```bash
# Disable smoothing to see raw policy behavior
python evaluate.py --no_smooth
```

---

## 📁 Project Structure

```
autonomous-driving-rl/
├── training.py              # Main training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
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
│   │   ├── reward_shaping.py
│   │   └── state_representations.py
│   │
│   └── utils/               # Utilities
│       └── logger.py
│
├── configs/                 # Configuration files
│   ├── algorithms/
│   └── experiments/
│
├── weights/                 # Trained models
│   ├── best_model.pth       # Best model for evaluate.py
│   └── final/               # All final models
│
├── results/
│   ├── logs/                # Training logs
│   ├── plots/               # Generated figures
│   └── tables/              # CSV results
│
└── scripts/                 # Utility scripts
    ├── evaluate_baselines.py
    ├── plot_results.py
    └── train_all.py
```

---

## 📊 Experiments

### Reward Shaping

I experimented with different reward modifications:

| Shaping   | Description                       | Result              |
| --------- | --------------------------------- | ------------------- |
| Default   | Original highway-env reward       | Baseline            |
| TTC       | Penalize low time-to-collision    | Similar performance |
| Smooth    | Penalize unnecessary lane changes | Lower performance   |
| Composite | TTC + Smooth combined             | Similar to default  |

<p align="center">
  <img src="results/plots/reward_shaping_comparison.png" alt="Reward Shaping" width="700"/>
</p>

**Finding**: Naive reward shaping can hurt performance. The default reward function is already well-tuned for this environment.

### Environment Variations

| Environment | Lanes | Vehicles | Mean Return |
| ----------- | ----- | -------- | ----------- |
| Easy        | 4     | 20       | 28.34       |
| Default     | 3     | 50       | 27.95       |
| Dense       | 3     | 100      | 27.73       |
| Hard        | 2     | 80       | 24.71       |

---

## 🔧 Configuration

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
```

### Evaluation Arguments

```bash
python evaluate.py --help

Options:
  --algorithm       Algorithm to evaluate
  --weights         Path to model weights
  --episodes        Number of evaluation episodes (default: 10)
  --no_render       Disable rendering
  --no_smooth       Disable action smoothing
  --smooth_window   Minimum steps between lane changes (default: 3)
```

---

## 📝 Course Project

This project was developed for the **Reinforcement Learning** course (2025/2026) at the University of Padova.

**Professor**: Gian Antonio Susto  
**Teaching Assistants**: Alberto Sinigaglia, Riccardo De Monte

---

## 🙏 Acknowledgments

- [Farama Foundation](https://farama.org/) for the highway-env environment
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for algorithm references
- Course instructors for guidance and support
