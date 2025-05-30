from abc import ABC, abstractmethod

from tuning.interfaces.agent_interface import AgentInterface


class EvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(self, agent: AgentInterface, n_episodes: int) -> float:
        raise NotImplementedError