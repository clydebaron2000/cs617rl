from tuning.config.optimization_config import OptimizationConfig, ParameterRange

SAC_CONFIG = OptimizationConfig(
    env_name= "BipedalWalker-v3",
    seed=0,
    max_timesteps=50_000,
    eval_interval=1000,
    n_eval_episodes=5,
    n_trials=10,
    timeout_hours=8,
    hyperparameters={
        'lr': ParameterRange(1e-5, 1e-3, log_scale=True),
        'gamma': ParameterRange(0.9, 0.999),
        'entropy_coef': ParameterRange(0.01, 1.0, log_scale=True),
        'batch_size': ParameterRange(64, 512, step=64),
        'tau': ParameterRange(0.001, 0.01, log_scale=True),
        'train_freq': ParameterRange(64,128,step=8),
        'gradient_steps': ParameterRange(16,52,step=4)
    }
)