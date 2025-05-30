from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Critic network that outputs Q-values for state-action pairs."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the critic network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Number of hidden units
        """
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            Tuple of (q1, q2) where q1 and q2 are the Q-values from the two networks
        """
        x = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)

class ValueCritic(nn.Module):
    """Value network that outputs state values."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize the value network.

        Args:
            state_dim: Dimension of the state space
            hidden_dim: Number of hidden units
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            State value
        """
        return self.net(state) 