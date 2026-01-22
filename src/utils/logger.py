"""Logging utilities for training metrics."""

import csv
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """Logger for training metrics with TensorBoard and CSV support."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_csv: bool = True
    ):
        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir)
        
        # CSV logging
        self.use_csv = use_csv
        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        self.csv_fields = None
    
    def log(self, metrics: Dict[str, float], step: int):
        """Log metrics to all enabled backends."""
        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        
        # CSV
        if self.use_csv:
            self._log_csv(metrics, step)
    
    def _log_csv(self, metrics: Dict[str, float], step: int):
        """Log metrics to CSV file."""
        row = {"step": step, **metrics}
        
        # Initialize CSV on first write
        if self.csv_writer is None:
            self.csv_fields = list(row.keys())
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
            self.csv_writer.writeheader()
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def log_hyperparameters(self, hparams: Dict):
        """Log hyperparameters."""
        if self.use_tensorboard:
            self.writer.add_hparams(hparams, {})
        
        # Save to file
        hparams_path = self.log_dir / "hparams.txt"
        with open(hparams_path, 'w') as f:
            for key, value in hparams.items():
                f.write(f"{key}: {value}\n")
    
    def close(self):
        """Close all logging backends."""
        if self.use_tensorboard:
            self.writer.close()
        if self.csv_file:
            self.csv_file.close()
