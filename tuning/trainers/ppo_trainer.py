# tuning/trainers/ppo_trainer.py
from typing import Dict, Any
import numpy as np
import gymnasium as gym

from tuning.config.optimization_config import OptimizationConfig
from tuning.trainers.base_trainer import BaseTrainer
from tuning.interfaces.agent_interface import AgentInterface
from ppo import PPOAgent
import torch


class PPOTrainer(BaseTrainer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf, self.rew_buf, self.val_buf, self.done_buf = [], [], [], [], [], [], [], []
    def clear_buff(self):
        self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf, self.rew_buf, self.val_buf, self.done_buf = [], [], [], [], [], [], [], []

    def create_agent(self, params: Dict[str, Any]) -> PPOAgent:
        state_dim, action_dim, max_action = self.env_manager.get_env_info()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return PPOAgent(state_dim, action_dim, max_action, device, **params)

    def train_step(self, agent: PPOAgent, env:gym.Env, state: np.ndarray, params: Dict[str, Any]) -> tuple:
        action, logp, value = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        self.obs_buf.append(state)
        self.act_buf.append(action)
        self.logp_buf.append(logp.item())
        self.rew_buf.append(reward)
        self.val_buf.append(value.item())
        self.done_buf.append(done)


        if len(self.obs_buf) >= params['n_steps'] or done:
            self.val_buf.append(0)
            self.adv_buf = agent.compute_advantages(self.rew_buf, self.val_buf, self.done_buf)
            self.ret_buf = self.adv_buf + self.val_buf[:-1]
            agent.update(self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf)

            self.clear_buff()

        if done:
            next_state, _ = env.reset()
        return next_state
