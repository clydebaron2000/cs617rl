import optuna
import os
import json
from datetime import datetime

from tuning.config.optimization_config import OptimizationConfig
from tuning.trainers.base_trainer import BaseTrainer


class OptimizationManager:
    def __init__(self, config: OptimizationConfig, trainer: BaseTrainer):
        self.config = config
        self.trainer = trainer
        self.save_dir = self.__create_save_directory()

    def __create_save_directory(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trainer_name = self.trainer.__class__.__name__.lower().replace('trainer', '')
        save_dir = f"results/{trainer_name}_optimization_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def optimize(self) -> optuna.Study:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            ),
            sampler=optuna.samplers.TPESampler(seed=self.config.seed)
        )

        study.optimize(
            self.trainer.train_agent,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_hours * 3600
        )

        self.__save_results(study)
        self.__print_results(study)

        return study

    def __save_results(self, study: optuna.Study) -> None:
        with open(os.path.join(self.save_dir, 'best_params.json'), 'w') as f:
            json.dump(study.best_params, f, indent=4)

        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })

        with open(os.path.join(self.save_dir, 'optimization_history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        stats = {
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }

        with open(os.path.join(self.save_dir, 'optimization_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    def __print_results(self, study: optuna.Study) -> None:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")