from typing import Dict, Any

import gymnasium as gym
import numpy as np
import torch

from a2c_gae import A2C
from tuning.config.optimization_config import OptimizationConfig
from tuning.trainers.base_trainer import BaseTrainer


class A2CTrainer:

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def create_agent(self, params: Dict[str, Any]) -> A2C:

        state_dim, action_dim, max_action = self.env_manager.get_env_info()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = A2C(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            learning_rate=params['learning_rate'],
            hidden_dim = params['hidden_dim'],
            log_std_init = params['log_std_init']
        )

        return agent

    def train_step(self, agent: A2C, env: gym.Env, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(float(done))

        if len(self.states) >= params['n_steps'] or done:
            agent.train(
                self.states, self.actions, self.rewards,
                self.next_states, self.dones,
                entropy_coef=params['entropy_coef'],
                gae_lambda=params['gae_lambda']
            )
            self.states, self.actions, self.rewards = [], [], []
            self.next_states, self.dones = [], []

        if done:
            next_state, _ = env.reset()
        return next_state
