from tuning.config.optimization_config import OptimizationConfig, ParameterRange

PPO_CONFIG = OptimizationConfig(
    env_name="BipedalWalker-v3",
    seed=0,
    max_timesteps=50_000,
    eval_interval=1000,
    n_eval_episodes=5,
    n_trials=10,
    timeout_hours=8,
    hyperparameters={
        # Learning rates
        'lr': ParameterRange(1e-5, 1e-3, log_scale=True),

        # Discount and GAE parameters
        'gamma': ParameterRange(0.9, 0.999),

        # PPO specific parameters
        'clip_ratio': ParameterRange(0.1, 0.3),

        # # Training parameters
        # 'batch_size': ParameterRange(64, 256, step=64),
        # 'steps_per_epoch': ParameterRange(100, 1600, step=100),
        # 'train_iters': ParameterRange(20, 400, step=10),
        #
        # # Normalization and clipping
        # 'target_kl': ParameterRange(0.00, 0.03),
        'n_steps': ParameterRange(8, 32, step=1),
    }
)
