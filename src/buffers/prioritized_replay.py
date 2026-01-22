"""Prioritized Experience Replay buffer."""

import numpy as np
from typing import Tuple


class SumTree:
    """Binary tree for efficient priority sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
    
    def update(self, idx: int, priority: float):
        """Update priority at leaf index."""
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get(self, value: float) -> Tuple[int, float]:
        """Sample index based on priority value."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]
    
    @property
    def total(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with proportional prioritization."""
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_decay_steps: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta_start  # Importance sampling weight
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay = (beta_end - beta_start) / beta_decay_steps
        
        self.tree = SumTree(capacity)
        self.min_priority = 1e-6
        self.max_priority = 1.0
        
        self.ptr = 0
        self.size = 0
        
        # Data storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """Store transition with max priority."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        # New transitions get max priority
        self.tree.update(self.ptr, self.max_priority ** self.alpha)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with priority-based probabilities."""
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx, priority = self.tree.get(value)
            
            indices[i] = idx
            priorities[i] = priority
        
        # Importance sampling weights
        probs = priorities / self.tree.total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(self.beta_end, self.beta + self.beta_decay)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + self.min_priority
        self.max_priority = max(self.max_priority, priorities.max())
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority ** self.alpha)
    
    def __len__(self) -> int:
        return self.size