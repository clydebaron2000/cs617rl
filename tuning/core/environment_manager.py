from typing import Tuple

import gymnasium as gym
from common.utils import setup_environment, get_env_info

from tuning.config.optimization_config import OptimizationConfig


class EnvironmentManager:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._env = None

    def create_environment(self) -> gym.Env:
        self._env = setup_environment(self.config.env_name, self.config.seed)
        return self._env

    def get_env_info(self) -> Tuple[int, int, float]:
        if self._env is None:
            self._env = self.create_environment()
        return get_env_info(self._env)

    def close(self) -> None:
        if self._env:
            self._env.close()
