from optuna.trial import Trial
from typing import Dict, Any

from tuning.config.optimization_config import OptimizationConfig


class ParameterSampler:
    def __init__(self, config: OptimizationConfig):
        self.config = config

    def sample_parameters(self, trial: Trial) -> Dict[str, Any]:
        params = {}
        for param_name, param_range in self.config.hyperparameters.items():
            if param_range.step is not None:
                params[param_name] = trial.suggest_int(
                    param_name,
                    int(param_range.min_val),
                    int(param_range.max_val),
                    step=param_range.step
                )
            else:
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_range.min_val,
                    param_range.max_val,
                    log=param_range.log_scale
                )
        return params