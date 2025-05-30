from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

@dataclass
class ParameterRange:
    min_val: float
    max_val: float
    log_scale:bool= False
    step:Optional[int]= None

@dataclass
class OptimizationConfig:
    env_name: str
    max_timesteps: int
    eval_interval: int
    n_eval_episodes: int
    n_trials: int
    timeout_hours: int
    hyperparameters: Dict[str, ParameterRange]
    seed: int = 0

