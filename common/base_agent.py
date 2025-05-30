import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

class BaseAgent:
    """Base class for reinforcement learning agents."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: torch.device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select an action from the current state."""
        raise NotImplementedError

    def train(self, *args, **kwargs) -> None:
        """Train the agent."""
        raise NotImplementedError

    def save(self, directory: str, name: str) -> None:
        """Save the agent's parameters."""
        raise NotImplementedError

    def load(self, directory: str, name: str) -> None:
        """Load the agent's parameters."""
        raise NotImplementedError

