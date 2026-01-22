# RL Autonomous Driving Project - Pipeline Completa

**Repository**: `autonomous-driving-rl`

## Directory Structure

```
autonomous-driving-rl/
│
├── configs/
│   ├── default.yaml                 # Config base condivisa
│   ├── experiments/
│   │   ├── state_representations.yaml
│   │   ├── reward_shaping.yaml
│   │   └── env_variations.yaml
│   └── algorithms/
│       ├── dqn.yaml
│       ├── double_dqn.yaml
│       ├── dueling_dqn.yaml
│       ├── d3qn.yaml
│       └── ppo.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Classe astratta base
│   │   ├── dqn_agent.py             # DQN vanilla
│   │   ├── double_dqn_agent.py      # Double DQN
│   │   ├── dueling_dqn_agent.py     # Dueling DQN
│   │   ├── d3qn_agent.py            # Double + Dueling + PER
│   │   └── ppo_agent.py             # PPO
│   │
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── mlp.py                   # MLP per Kinematics
│   │   ├── cnn.py                   # CNN per OccupancyGrid/Images
│   │   ├── dueling_network.py       # Dueling architecture
│   │   └── actor_critic.py          # Per PPO
│   │
│   ├── buffers/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py         # Standard experience replay
│   │   ├── prioritized_replay.py    # PER con SumTree
│   │   └── rollout_buffer.py        # Per PPO
│   │
│   ├── env/
│   │   ├── __init__.py
│   │   ├── wrappers.py              # Custom env wrappers
│   │   ├── reward_shaping.py        # Custom reward functions
│   │   └── state_representations.py # Diverse rappresentazioni stato
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                # Logging + TensorBoard
│       ├── config.py                # Config loader
│       ├── seed.py                  # Seed management
│       └── metrics.py               # Metriche di valutazione
│
├── baselines/
│   ├── __init__.py
│   ├── heuristic_ttc.py             # Baseline TTC-based
│   ├── heuristic_simple.py          # Baseline naive (always right + fast)
│   └── manual_control.py            # Manual control modificato
│
├── experiments/
│   ├── run_single_algorithm.py      # Training singolo algoritmo
│   ├── run_algorithm_comparison.py  # Confronto tutti gli algoritmi
│   ├── run_state_repr_comparison.py # Confronto state representations
│   ├── run_reward_shaping.py        # Esperimenti reward shaping
│   └── run_env_variations.py        # Esperimenti config ambiente
│
├── scripts/
│   ├── train.py                     # Script training principale
│   ├── evaluate.py                  # Script evaluation principale
│   ├── plot_results.py              # Genera plot per report
│   └── generate_tables.py           # Genera tabelle per report
│
├── results/
│   ├── logs/                        # TensorBoard logs
│   ├── checkpoints/                 # Model checkpoints durante training
│   ├── plots/                       # Plot generati
│   └── tables/                      # Tabelle CSV
│
├── weights/
│   └── best_model.pth               # Modello finale ottimale
│
├── report/
│   ├── main.tex                     # Report LaTeX
│   ├── figures/                     # Figure per report
│   └── bibliography.bib             # Bibliografia
│
├── tests/                           # Unit tests (opzionale ma utile)
│   ├── test_agents.py
│   ├── test_buffers.py
│   └── test_env.py
│
├── notebooks/                       # Jupyter notebooks per analisi
│   └── analysis.ipynb
│
├── training.py                      # Entry point training (richiesto)
├── evaluate.py                      # Entry point evaluation (richiesto)
├── your_baseline.py                 # Entry point baseline (richiesto)
├── requirements.txt                 # Dipendenze
├── README.md                        # Documentazione
└── PROJECT_PIPELINE.md              # Questo file
```

---

## Pipeline di Sviluppo

### FASE 0: Setup Iniziale [Giorno 1]
```
□ 0.1 Creare struttura directory
□ 0.2 Setup ambiente conda/venv
□ 0.3 Installare dipendenze (highway-env, torch, etc.)
□ 0.4 Verificare che highway-env funzioni
□ 0.5 Creare file requirements.txt
□ 0.6 Setup logging e config system
```

**Deliverable**: Ambiente funzionante, env che gira

---

### FASE 1: Infrastruttura Core [Giorni 2-4]
```
□ 1.1 Implementare src/utils/config.py (YAML loader)
□ 1.2 Implementare src/utils/seed.py (seed management)
□ 1.3 Implementare src/utils/logger.py (TensorBoard + CSV)
□ 1.4 Implementare src/utils/metrics.py (metriche evaluation)
□ 1.5 Implementare src/buffers/replay_buffer.py
□ 1.6 Implementare src/networks/mlp.py
□ 1.7 Implementare src/agents/base_agent.py (classe astratta)
□ 1.8 Creare configs/default.yaml
```

**Deliverable**: Infrastruttura pronta per gli agenti

---

### FASE 2: DQN Base [Giorni 5-7]
```
□ 2.1 Implementare src/agents/dqn_agent.py
□ 2.2 Creare configs/algorithms/dqn.yaml
□ 2.3 Implementare training loop base in scripts/train.py
□ 2.4 Implementare evaluation loop in scripts/evaluate.py
□ 2.5 Test training completo (~50k steps)
□ 2.6 Verificare che l'agente impari (reward crescente)
□ 2.7 Debug e fix eventuali problemi
```

**Deliverable**: DQN funzionante, baseline RL

---

### FASE 3: Baseline Euristiche [Giorno 8]
```
□ 3.1 Implementare baselines/heuristic_ttc.py (TTC-based)
□ 3.2 Implementare baselines/heuristic_simple.py (naive)
□ 3.3 Modificare baselines/manual_control.py (logging)
□ 3.4 Valutare tutte le baseline (10 episodi)
□ 3.5 Salvare metriche baseline per confronto
□ 3.6 Creare your_baseline.py entry point
```

**Deliverable**: Baseline pronte, metriche salvate

---

### FASE 4: Estensioni DQN [Giorni 9-12]
```
□ 4.1 Implementare src/agents/double_dqn_agent.py
□ 4.2 Implementare src/networks/dueling_network.py
□ 4.3 Implementare src/agents/dueling_dqn_agent.py
□ 4.4 Implementare src/buffers/prioritized_replay.py (SumTree)
□ 4.5 Implementare src/agents/d3qn_agent.py (Double+Dueling+PER)
□ 4.6 Creare config files per ogni variante
□ 4.7 Test training per ogni variante
□ 4.8 Confronto preliminare performance
```

**Deliverable**: 4 varianti DQN funzionanti

---

### FASE 5: PPO [Giorni 13-16]
```
□ 5.1 Implementare src/networks/actor_critic.py
□ 5.2 Implementare src/buffers/rollout_buffer.py
□ 5.3 Implementare src/agents/ppo_agent.py
□ 5.4 Creare configs/algorithms/ppo.yaml
□ 5.5 Test training PPO
□ 5.6 Tuning iperparametri PPO
□ 5.7 Confronto PPO vs DQN family
```

**Deliverable**: PPO funzionante, 5 algoritmi totali

---

### FASE 6: Bonus - State Representations [Giorni 17-19]
```
□ 6.1 Implementare src/env/state_representations.py
       - Kinematics (default, 5x5 → 25)
       - OccupancyGrid (griglia 2D)
       - Grayscale Image (immagine)
□ 6.2 Implementare src/networks/cnn.py per OccupancyGrid/Images
□ 6.3 Creare src/env/wrappers.py per cambio rappresentazione
□ 6.4 Training DQN/PPO con ogni rappresentazione
□ 6.5 Confronto e analisi risultati
□ 6.6 Salvare metriche per report
```

**Deliverable**: 3 state representations testate

---

### FASE 7: Bonus - Reward Shaping [Giorni 20-21]
```
□ 7.1 Implementare src/env/reward_shaping.py
       - Default reward
       - TTC penalty reward
       - Smoothness reward (penalità lane change)
       - Composite reward
□ 7.2 Training con ogni reward function
□ 7.3 Confronto e analisi risultati
□ 7.4 Identificare reward function ottimale
```

**Deliverable**: Reward shaping testato

---

### FASE 8: Bonus - Environment Variations [Giorno 22]
```
□ 8.1 Definire variazioni da testare:
       - vehicles_density: [0.5, 1.0, 1.5, 2.0]
       - lanes_count: [2, 3, 4]
       - vehicles_count: [20, 50, 100]
□ 8.2 Training su subset significativo di combinazioni
□ 8.3 Analisi robustezza dell'agente
□ 8.4 Identificare configurazione ottimale
```

**Deliverable**: Analisi robustezza completata

---

### FASE 9: Training Finale & Selection [Giorni 23-25]
```
□ 9.1 Analizzare tutti i risultati raccolti
□ 9.2 Identificare combinazione ottimale:
       - Miglior algoritmo
       - Miglior state representation
       - Miglior reward function
       - Miglior env config
□ 9.3 Training finale esteso (100k+ steps)
□ 9.4 Hyperparameter tuning finale
□ 9.5 Salvare best_model.pth
□ 9.6 Validazione su 100 episodi
```

**Deliverable**: Modello finale ottimale

---

### FASE 10: Evaluation & Plots [Giorni 26-27]
```
□ 10.1 Eseguire evaluation completa tutti gli algoritmi
□ 10.2 Eseguire evaluation baseline
□ 10.3 Generare learning curves
□ 10.4 Generare comparison plots
□ 10.5 Generare tabelle risultati
□ 10.6 Creare scripts/plot_results.py
□ 10.7 Esportare figure per report
```

**Deliverable**: Tutti i plot e tabelle pronti

---

### FASE 11: Report LaTeX [Giorni 28-30]
```
□ 11.1 Setup template LaTeX fornito
□ 11.2 Scrivere sezione: Critical Issues
□ 11.3 Scrivere sezione: Baseline Description
□ 11.4 Scrivere sezione: RL Algorithms
□ 11.5 Scrivere sezione: Results (con plot)
□ 11.6 Scrivere sezione: Discussion
□ 11.7 Scrivere sezione: Bonus Experiments
□ 11.8 Review e polish
□ 11.9 Compilare PDF finale
```

**Deliverable**: Report completo 6 pagine

---

### FASE 12: Submission Package [Giorno 31]
```
□ 12.1 Verificare che `python evaluate.py` funzioni
□ 12.2 Verificare che `python training.py` funzioni
□ 12.3 Verificare che `python your_baseline.py` funzioni
□ 12.4 Pulire codice (rimuovere debug, commenti inutili)
□ 12.5 Verificare requirements.txt completo
□ 12.6 Creare ZIP finale
□ 12.7 Test ZIP su ambiente pulito
□ 12.8 Submit su Moodle
```

**Deliverable**: Submission completata ✓

---

## Dipendenze tra Fasi

```
FASE 0 (Setup)
    │
    ▼
FASE 1 (Infrastruttura)
    │
    ├──────────────────┐
    ▼                  ▼
FASE 2 (DQN)      FASE 3 (Baselines)
    │
    ▼
FASE 4 (DQN Extensions)
    │
    ▼
FASE 5 (PPO)
    │
    ├─────────┬─────────┐
    ▼         ▼         ▼
FASE 6    FASE 7    FASE 8
(State)   (Reward)  (Env)
    │         │         │
    └─────────┴─────────┘
              │
              ▼
         FASE 9 (Training Finale)
              │
              ▼
         FASE 10 (Plots)
              │
              ▼
         FASE 11 (Report)
              │
              ▼
         FASE 12 (Submission)
```

---

## Checkpoint di Verifica

| Checkpoint | Fase | Criterio di Successo |
|------------|------|----------------------|
| CP1 | Fine Fase 2 | DQN converge, reward > baseline random |
| CP2 | Fine Fase 5 | 5 algoritmi funzionanti |
| CP3 | Fine Fase 8 | Tutti i bonus implementati |
| CP4 | Fine Fase 10 | Plot pronti per report |
| CP5 | Fine Fase 12 | `python evaluate.py` gira senza errori |

---

## Stima Effort per Algoritmo

| Algoritmo | Linee Codice (stima) | Difficoltà | Tempo |
|-----------|----------------------|------------|-------|
| DQN | ~150 | ⭐ | 2-3 giorni |
| Double DQN | ~20 (delta) | ⭐ | 0.5 giorni |
| Dueling DQN | ~80 (network) | ⭐⭐ | 1 giorno |
| D3QN + PER | ~200 (buffer PER) | ⭐⭐⭐ | 2-3 giorni |
| PPO | ~250 | ⭐⭐⭐⭐ | 3-4 giorni |

---

## Note Importanti

1. **Backup frequenti**: Git commit ad ogni fase completata
2. **Testing incrementale**: Non procedere se la fase precedente ha bug
3. **Logging consistente**: Usare sempre il logger per tracciare esperimenti
4. **Seed fisso**: Mai dimenticare seed=0 per riproducibilità
5. **Early stopping**: Se un algoritmo non converge dopo N steps, investigare prima di procedere

---

## Comandi Utili

```bash
# Training singolo algoritmo
python scripts/train.py --algorithm dqn --config configs/algorithms/dqn.yaml

# Evaluation
python scripts/evaluate.py --weights weights/best_model.pth

# Run baseline
python your_baseline.py

# Genera plot
python scripts/plot_results.py --results_dir results/

# Training comparison (tutti gli algoritmi)
python experiments/run_algorithm_comparison.py
```

---

## Prossimo Passo

**Iniziare FASE 0**: Setup iniziale e creazione struttura directory.

Confermi di procedere con la Fase 0?
