# Autonomous Driving RL

Deep Reinforcement Learning agents for autonomous highway driving.

**Course**: Reinforcement Learning 2025/2026 - University of Padova  
**Environment**: [highway-env](https://github.com/Farama-Foundation/HighwayEnv)

---

## 🎯 Project Goals

Train RL agents to safely and efficiently drive through a highway with other vehicles.

### Implemented Algorithms
- **DQN** - Deep Q-Network (baseline)
- **Double DQN** - Reduces overestimation bias
- **Dueling DQN** - Separate value and advantage streams
- **D3QN** - Double + Dueling + Prioritized Experience Replay
- **PPO** - Proximal Policy Optimization

### Bonus Experiments
- Multiple state representations (Kinematics, OccupancyGrid, Grayscale)
- Custom reward shaping (TTC penalty, smoothness bonus)
- Environment configuration variations

---

## 📁 Project Structure

```
autonomous-driving-rl/
├── src/                    # Core source code
│   ├── agents/             # RL agents implementations
│   ├── networks/           # Neural network architectures
│   ├── buffers/            # Experience replay buffers
│   ├── env/                # Environment wrappers and utilities
│   └── utils/              # Logging, config, metrics
├── baselines/              # Heuristic baselines
├── configs/                # YAML configuration files
├── experiments/            # Experiment scripts
├── scripts/                # Training and evaluation scripts
├── results/                # Logs, checkpoints, plots
├── weights/                # Trained model weights
├── report/                 # LaTeX report
├── training.py             # Main training entry point
├── evaluate.py             # Main evaluation entry point
└── baseline.py        # Baseline entry point
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/autonomous-driving-rl.git
cd autonomous-driving-rl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration (DQN)
python training.py

# Train specific algorithm
python scripts/train.py --algorithm ppo --config configs/algorithms/ppo.yaml
```

### Evaluation

```bash
# Evaluate trained agent
python evaluate.py

# Run baseline
python your_baseline.py
```

---

## 🎮 Environment

### State Space
- `5 x 5` matrix: 5 vehicles × 5 features
- Features: presence, x position, y position, x velocity, y velocity
- All values normalized w.r.t. ego-vehicle

### Action Space (Discrete)
| Action | Description |
|--------|-------------|
| 0 | Change lane left |
| 1 | Idle |
| 2 | Change lane right |
| 3 | Go faster |
| 4 | Go slower |

### Reward Function
- ✅ Bonus for high velocity
- ✅ Bonus for rightmost lane
- ❌ Penalty for collisions

---

## 📊 Results

See `results/` directory for:
- Learning curves
- Algorithm comparisons
- Ablation studies

---

## 📝 Report

The 6-page LaTeX report is available in `report/main.pdf`.

---

## 🔧 Configuration

All configurations are in YAML format under `configs/`:

```yaml
# configs/default.yaml
env:
  name: highway-fast-v0
  lanes_count: 3
  vehicles_count: 50
  duration: 40

training:
  max_steps: 100000
  batch_size: 64
  learning_rate: 0.0005
  gamma: 0.99

seed: 0
```

---

## 📚 References

- [highway-env Documentation](https://highway-env.farama.org/)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

## 📄 License

This project is for educational purposes (RL Course 2025/2026).
