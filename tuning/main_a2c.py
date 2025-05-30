import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.core.optimization_manager import OptimizationManager
from tuning.trainers.a2c_trainer import A2CTrainer
from tuning.config.a2c_config import A2C_CONFIG


def main():
    trainer = A2CTrainer(A2C_CONFIG)
    optimizer = OptimizationManager(A2C_CONFIG, trainer)
    study = optimizer.optimize()


if __name__ == "__main__":
    main()
