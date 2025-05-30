import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import time

from common.utils import plot_metrics, print_episode_info, run_evaluation, save_expt_metadata


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 128,log_std_init=0.0):
        super().__init__()
        self.activation = torch.tanh

        self.affine_layers = nn.ModuleList([
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std_init)

    def forward(self, state):
        x = state
        for layer in self.affine_layers:
            x = self.activation(layer(x))
        mean = self.action_mean(x)
        std = torch.exp(self.action_log_std.expand_as(mean))
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()
        action_tanh = torch.tanh(action)
        log_prob = dist.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        return action_tanh, log_prob.sum(1, keepdim=True)


class Value(nn.Module):
    def __init__(self, state_dim,hidden_dim = 128):
        super().__init__()
        self.activation = torch.tanh

        self.affine_layers = nn.ModuleList([
            nn.Linear(state_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

        self.value_head = nn.Linear(hidden_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, state):
        x = state
        for layer in self.affine_layers:
            x = self.activation(layer(x))
        return self.value_head(x).squeeze(-1)


class A2C:
    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim = 128, log_std_init = 0.0 ,learning_rate=3e-4):
        self.actor = Policy(state_dim, action_dim, hidden_dim=hidden_dim,log_std_init=log_std_init).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Value(state_dim,hidden_dim=hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.max_action = max_action
        self.device = device

    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, gae_lambda=0.9):
        advantages = []
        returns = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages * 0
        return advantages, returns

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.cpu().data.numpy().flatten()
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99, gae_lambda=0.9, entropy_coef=0.01):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        values = self.critic(states).squeeze(-1)
        if values.ndim == 0:
            values = values.unsqueeze(0)
        next_values = self.critic(next_states).squeeze(-1)
        if next_values.ndim == 0:
            next_values = next_values.unsqueeze(0)

        advantages, returns = self.compute_gae(
            rewards.cpu().numpy(),
            values.detach().cpu().numpy(),
            next_values.detach().cpu().numpy(),
            dones.cpu().numpy(),
            gamma,
            gae_lambda
        )

        advantages = torch.clamp(advantages, -10, 10)

        _, log_prob = self.actor.sample(states)
        actor_loss = -(log_prob * advantages.unsqueeze(1)).mean()
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        entropy = dist.entropy().sum(1).mean()
        actor_loss -= entropy_coef * entropy

        critic_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f'{directory}/a2c_actor_{name}.pth')
        torch.save(self.critic.state_dict(), f'{directory}/a2c_critic_{name}.pth')

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f'{directory}/a2c_actor_{name}.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/a2c_critic_{name}.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="BipedalWalker-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--n_steps", default=8, type=int)
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--entropy_coef", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--gae_lambda", default=0.9, type=float)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--log_std_init", type=float, default=0.0)
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--model_path", type=str, help="Path to the model to evaluate")
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    args = parser.parse_args()

    save_dir = f"results/a2c_gae_{args.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(args.env, render_mode="rgb_array")
    if args.save_video:
        env = RecordVideo(env, f"{save_dir}/videos", episode_trigger=lambda x: x % 50 == 0)

    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = A2C(state_dim, action_dim, max_action, device, log_std_init = args.log_std_init,hidden_dim=args.hidden_dim,learning_rate=args.learning_rate)
    if args.evaluate:
        run_evaluation(agent, env, args)
        return

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    # Initialize lists to track training metrics
    all_rewards = []
    all_episode_lengths = []

    # Initialize lists to track losses
    actor_losses = []
    critic_losses = []
    entropies = []

    total_training_start = time.time()
    episode_start = time.time()

    for t in range(args.max_timesteps):
        episode_timesteps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(float(done))

        state = next_state
        episode_reward += reward

        if len(states) >= args.n_steps or done:
            train_info = agent.train(
                states, actions, rewards, next_states, dones,
                entropy_coef=args.entropy_coef,
                gae_lambda=args.gae_lambda
            )
            # Save losses
            actor_losses.append(float(train_info['actor_loss']))
            critic_losses.append(float(train_info['critic_loss']))
            entropies.append(float(train_info['entropy']))

            states, actions, rewards, next_states, dones = [], [], [], [], []

            if t % 100 == 0:
                print(f"Step {t}, Actor Loss: {train_info['actor_loss']:.3f}, Critic Loss: {train_info['critic_loss']:.3f}, Entropy: {train_info['entropy']:.3f}")

        if done:
            print_episode_info(t+1, episode_num, episode_timesteps, episode_reward, time.time() - episode_start)
            # Store episode metrics
            all_rewards.append(float(episode_reward))
            all_episode_lengths.append(int(episode_timesteps))
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            episode_start = time.time()

        if (t + 1) % args.save_freq == 0:
            agent.save(save_dir, f"step_{t+1}")
    
    # Calculate and print total training time
    total_training_time = time.time() - total_training_start
    print(f"\nTotal training time: {total_training_time:.2f} seconds")

    # Save training curves and metadata
    plot_metrics(
        data=[all_rewards, all_episode_lengths],
        data_labels=["Episode Reward", "Episode Length"],
        x_label="Episodes",
        sma_window_size=100,
        save_dir=save_dir,
        img_name="training_curves.png",
    )
    plot_metrics(
        data=[actor_losses, critic_losses, entropies],
        data_labels=["Actor Loss", "Critic Loss", "Entropy"],
        x_label="Steps",
        sma_window_size=10_000,
        save_dir=save_dir,
        img_name="convergence_metrics.png",
    )
    save_expt_metadata(
        save_dir=save_dir,
        hyperparameters= vars(args),
        episode_rewards=all_rewards,
        episode_lengths=all_episode_lengths,
        total_training_time=total_training_time,
        convergence_metrics={
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "entropies": entropies,
        }
    )
    env.close()


if __name__ == "__main__":
    main()