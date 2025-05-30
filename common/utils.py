import argparse
import json
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from datetime import datetime
from gymnasium.wrappers import RecordVideo

from common.base_agent import BaseAgent


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def plot_metrics(
    data: List[List[float]],
    data_labels: List[str],
    x_label: str,
    save_dir: str,
    img_name: str,
    sma_window_size: int
) -> None:

    n_metrics = len(data)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, (metric_data, label) in enumerate(zip(data, data_labels)):
        ax = axes[i]
        steps = np.arange(len(metric_data))

        ax.plot(steps, metric_data, "b-", alpha=0.6, label=label)

        if len(metric_data) >= sma_window_size:
            moving_avg = np.convolve(metric_data, np.ones(sma_window_size) / sma_window_size, mode="valid")
            ax.plot(
                steps[sma_window_size - 1 :],
                moving_avg,
                "r-",
                label=f"Moving Average ({sma_window_size} {x_label.lower()})",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
        ax.set_title(f"{label} Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, img_name), dpi=300, bbox_inches="tight"
    )
    plt.close()

def evaluate_policy(
    agent: BaseAgent,
    env: gym.Env,
    num_episodes: int = 100,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Union[float, List[float], List[int]]]:
    """
    Evaluate a trained policy over multiple episodes and calculate reward statistics.

    Args:
        agent: The trained agent
        env: The environment to evaluate in
        num_episodes: Number of episodes to run
        save_dir: Directory to save evaluation results

    Returns:
        dict: Dictionary containing evaluation statistics
    """
    all_rewards: List[float] = []
    all_lengths: List[int] = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)
            episode_length += 1

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}"
            )

    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))
    min_reward = float(np.min(all_rewards))
    max_reward = float(np.max(all_rewards))

    mean_length = float(np.mean(all_lengths))
    std_length = float(np.std(all_lengths))
    min_length = int(np.min(all_lengths))
    max_length = int(np.max(all_lengths))

    if save_dir is not None:
        eval_dir = os.path.join(save_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.hist(all_rewards, bins=20, alpha=0.75, color="blue")
        ax1.axvline(
            mean_reward,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_reward:.2f} ± {std_reward:.2f}",
        )
        ax1.set_title("Reward Distribution")
        ax1.set_xlabel("Episode Reward")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.hist(all_lengths, bins=20, alpha=0.75, color="green")
        ax2.axvline(
            mean_length,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_length:.2f} ± {std_length:.2f}",
        )
        ax2.set_title("Episode Length Distribution")
        ax2.set_xlabel("Episode Length")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(eval_dir, "distributions.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
    if verbose:
        print("\nEvaluation Summary:")
        print(f"Number of episodes: {num_episodes}")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Min reward: {min_reward:.2f}")
        print(f"Max reward: {max_reward:.2f}")
        print(f"Mean episode length: {mean_length:.2f} ± {std_length:.2f}")
        print(f"Min episode length: {min_length}")
        print(f"Max episode length: {max_length}")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_length": mean_length,
        "std_length": std_length,
        "min_length": min_length,
        "max_length": max_length,
        "all_rewards": all_rewards,
        "all_lengths": all_lengths,
    }


class EvalWrapper(gym.Wrapper):
    """Wrapper that uses evaluation mode for actions."""

    def __init__(self, env: gym.Env, agent: BaseAgent):
        super().__init__(env)
        self.agent = agent
        self.last_observation: Optional[np.ndarray] = None

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.last_observation is None:
            raise ValueError("Environment must be reset before stepping")
        eval_action = self.agent.select_action(self.last_observation, evaluate=True)
        next_obs, reward, terminated, truncated, info = self.env.step(eval_action)
        self.last_observation = next_obs
        return next_obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset(**kwargs)
        self.last_observation = state
        return state, info


def setup_environment(
    env_name: str, seed: int, render_mode: str = "rgb_array"
) -> gym.Env:
    """
    Set up the gym environment with proper seeding.

    Args:
        env_name: Name of the environment
        seed: Random seed
        render_mode: Render mode for the environment

    Returns:
        gym.Env: Configured environment
    """
    env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return env


def get_env_info(env: gym.Env) -> Tuple[int, int, float]:
    """
    Get environment dimensions and action bounds.

    Args:
        env: Gym environment

    Returns:
        Tuple containing (state_dim, action_dim, max_action)
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    return state_dim, action_dim, max_action


def setup_save_directory(algorithm: str, env_name: str) -> str:
    """
    Create and return a directory for saving results.
    """
    save_dir = (
        f"results/{algorithm}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def setup_video_recording(env: gym.Env, save_dir: str) -> gym.Env:
    """
    Set up video recording for the environment.

    Args:
        env: Gym environment
        save_dir: Directory to save videos

    Returns:
        gym.Env: Environment wrapped with video recording
    """
    return RecordVideo(env, f"{save_dir}/videos", episode_trigger=lambda x: x % 50 == 0)


def print_episode_info(
    total_steps: int,
    episode_num: int,
    episode_steps: int,
    episode_reward: float,
    episode_time: float,
) -> None:
    """
    Print episode information in a formatted way.
    """
    print(
        f"Total steps: {total_steps:7d} | "
        f"Episode num: {episode_num+1:4d} | "
        f"Episode steps: {episode_steps:4d} | "
        f"Reward: {episode_reward:8.3f} | "
        f"Time: {episode_time:6.2f}s"
    )


def load_model_for_evaluation(
    agent: BaseAgent,
    model_path: str,
) -> None:
    """
    Load a trained model for evaluation.

    Args:
        agent: The agent to load the model into
        model_path: Path to the model file
        save_dir: Directory to save evaluation results
    """
    if model_path is None:
        raise ValueError("Model path must be provided for evaluation mode")

    model_dir = os.path.dirname(model_path)
    step_name = os.path.basename(model_path).replace(".pth", "").split("_")[-1]
    model_identifier = f"step_{step_name}"

    agent.load(directory=model_dir, name=model_identifier)
    print(f"Loaded model from {model_path}")


def run_evaluation(agent: BaseAgent, env: gym.Env, args: argparse.Namespace) -> None:
    load_model_for_evaluation(agent, args.model_path)
    evaluate_policy(
        agent,
        env,
        num_episodes=args.eval_episodes,
        save_dir=os.path.dirname(args.model_path),
    )
    env.close()


def save_expt_metadata(
    save_dir: str,
    hyperparameters: Dict[str, Any],
    episode_rewards: List[float],
    episode_lengths: List[int],
    total_training_time: float,
    convergence_metrics: Dict[str, List[float]],
) -> None:
    with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
        json.dump(
            {
                "total_training_time": float(total_training_time),
                "hyperparameters": hyperparameters,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "convergence_metrics": convergence_metrics,
            },
            f,
            indent=4,
        )
