import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.config.sac_config import SAC_CONFIG
from tuning.core.optimization_manager import OptimizationManager
from tuning.trainers.sac_trainer import SACTrainer


def main():
    trainer = SACTrainer(SAC_CONFIG)
    optimizer = OptimizationManager(SAC_CONFIG, trainer)
    study = optimizer.optimize()


if __name__ == "__main__":
    main()
