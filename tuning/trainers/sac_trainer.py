from typing import Dict, Any
import gymnasium as gym
import numpy as np
import torch
from sac_biped_walker import SAC
from common.replay_buffer import ReplayBuffer
from tuning.config.optimization_config import OptimizationConfig
from tuning.trainers.base_trainer import BaseTrainer


class SACTrainer(BaseTrainer):
    def __init__(self, config: OptimizationConfig, learning_start: int = 10_000):
        super().__init__(config)
        self.replay_buffer = ReplayBuffer(1_000_000)
        self.learning_start = learning_start
        self.t = 0

    def create_agent(self, params: Dict[str, Any]) -> SAC:

        state_dim, action_dim, max_action = self.env_manager.get_env_info()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = SAC(state_dim, action_dim, max_action, device)
        agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=params["lr"])
        agent.critic_optimizer = torch.optim.Adam(
            agent.critic.parameters(), lr=params["lr"]
        )
        agent.alpha_optimizer = torch.optim.Adam([agent.log_alpha], lr=params["lr"])

        return agent

    def train_step(self, agent: SAC, env: gym.Env, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        self.replay_buffer.push(state, action, reward, next_state, float(done))
        if len(self.replay_buffer) > self.learning_start and self.t % params["train_freq"] == 0:
            for _ in range(params["gradient_steps"]):
                agent.train(
                    self.replay_buffer, params["batch_size"], params["gamma"], params["tau"]
                )
        if done:
            next_state, _ = env.reset()
        self.t+=1
        return next_state