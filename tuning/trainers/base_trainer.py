from abc import ABC, abstractmethod
from typing import Dict, Any

import optuna
from optuna import Trial

from tuning.config.optimization_config import OptimizationConfig
from tuning.core.environment_manager import EnvironmentManager
from tuning.core.parameter_sampler import ParameterSampler
from tuning.evaluators.standard_evaluator import StandardEvaluator
from tuning.interfaces.agent_interface import AgentInterface


class BaseTrainer(ABC):
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.env_manager = EnvironmentManager(config)
        self.parameter_sampler = ParameterSampler(config)

    @abstractmethod
    def create_agent(self, params: Dict[str, Any]) -> AgentInterface:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, agent: AgentInterface, *args, **kwargs) -> None:
        raise NotImplementedError

    def train_agent(self, trial: Trial) -> float:
        params = self.parameter_sampler.sample_parameters(trial)
        env = self.env_manager.create_environment()
        agent = self.create_agent(params)
        evaluator = StandardEvaluator(env)
        state, _ = env.reset()
        eval_rewards = []

        for t in range(self.config.max_timesteps):
            print(f"Steps taken in current trial: {t}/{self.config.max_timesteps}", end="\r")
            state = self.train_step(agent, env, state, params)

            if (t + 1) % self.config.eval_interval == 0:
                eval_reward = evaluator.evaluate(agent, self.config.n_eval_episodes)
                eval_rewards.append(eval_reward)
                trial.report(float(eval_reward), t + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        self.env_manager.close()
        return float(max(eval_rewards)) if eval_rewards else float(-np.inf)