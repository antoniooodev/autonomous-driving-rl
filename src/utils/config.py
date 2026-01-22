"""Configuration loader utility."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Configuration wrapper with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"Config({self.to_dict()})"


def get_config(config_path: str, overrides: Optional[Dict] = None) -> Config:
    """Load config with optional overrides."""
    config = load_config(config_path)
    if overrides:
        config = merge_configs(config, overrides)
    return Config(config)
