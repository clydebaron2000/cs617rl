import numpy as np
import gymnasium as gym

from tuning.interfaces.agent_interface import AgentInterface
from tuning.interfaces.evaluator_interface import EvaluatorInterface


class StandardEvaluator(EvaluatorInterface):
    def __init__(self, env: gym.Env):
        self.env = env

    def evaluate(self, agent: AgentInterface, n_episodes: int) -> float:
        eval_rewards = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            eval_rewards.append(episode_reward)

        return float(np.mean(eval_rewards))