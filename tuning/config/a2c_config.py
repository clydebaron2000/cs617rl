from tuning.config.optimization_config import OptimizationConfig, ParameterRange

A2C_CONFIG = OptimizationConfig(
    env_name="BipedalWalker-v3",
    seed=0,
    max_timesteps=100_000,
    eval_interval=50_000,
    n_eval_episodes=5,
    n_trials=1,
    timeout_hours=8,
    hyperparameters={
        'learning_rate': ParameterRange(1e-5, 1e-3, log_scale=True),
        'n_steps': ParameterRange(8, 32, step=1),
        'entropy_coef': ParameterRange(1e-5, 5e-3, log_scale=True),
        'gae_lambda': ParameterRange(0.8, 0.99),
        'hidden_dim': ParameterRange(64, 256, step=1),
        'log_std_init': ParameterRange(-1.0, 1.0),
    }
)