# main.py
import optuna

from tuning.config.ppo_config import PPO_CONFIG
from tuning.trainers.ppo_trainer import PPOTrainer


def objective(trial: optuna.Trial) -> float:
    # Create trainer
    trainer = PPOTrainer(PPO_CONFIG)

    # Train and return mean reward
    return trainer.train_agent(trial)


def main():
    # Create study
    study = optuna.create_study(study_name="PPO_GAE", direction="maximize")

    # Optimize
    study.optimize(objective, n_trials=50)

    # Print results
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
