from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import torch


class AgentInterface(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def configure_hyperparameters(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError