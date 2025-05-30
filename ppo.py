import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Normal
import matplotlib.pyplot as plt

from tuning.interfaces.agent_interface import AgentInterface


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), log_std_init=-0.5):
        super().__init__()
        self.policy_net = self._mlp(obs_dim, hidden_sizes, act_dim)
        self.value_net = self._mlp(obs_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _mlp(self, input_dim, hidden_sizes, output_dim):
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def step(self, obs):
        with torch.no_grad():
            obs = obs.to(next(self.parameters()).device)
            mean = self.policy_net(obs)
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value = self.value_net(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        obs = obs.to(next(self.parameters()).device)
        actions = actions.to(next(self.parameters()).device)
        mean = self.policy_net(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        values = self.value_net(obs).squeeze(-1)
        return log_probs, entropy, values


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        device,
        hidden_sizes=(64, 64),
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=80,
        target_kl=0.01,
        gamma=0.99,
        lam=0.95,
        steps_per_epoch=4000,
        batch_size=64,
            **kwargs
    ):
        self.actor_critic = MLPActorCritic(obs_dim, act_dim, hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.device = device
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.gamma = gamma
        self.lam = lam
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.act_limit = torch.tensor(act_limit).to(self.device)
        self.reward_avg = []
    def to_tensor(self,x):
        return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)

    def compute_advantages(self, rewards, values, dones):
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * (1 - dones[t]) * lastgaelam
        return adv

    def update(self, obs_buf, act_buf, adv_buf, ret_buf, logp_old_buf):
        obs_buf = self.to_tensor(obs_buf)
        act_buf = self.to_tensor(act_buf)
        adv_buf = self.to_tensor(adv_buf)
        ret_buf = self.to_tensor(ret_buf)
        logp_old_buf = self.to_tensor(logp_old_buf)

        for _ in range(self.train_iters):
            idx = np.random.permutation(len(obs_buf))
            for start in range(0, len(obs_buf), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                logp, entropy, values = self.actor_critic.evaluate(obs_buf[batch_idx], act_buf[batch_idx])
                ratio = torch.exp(logp - logp_old_buf[batch_idx])
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                loss_pi = -torch.min(ratio * adv_buf[batch_idx], clipped * adv_buf[batch_idx]).mean()

                loss_v = ((values - ret_buf[batch_idx]) ** 2).mean()
                loss_entropy = entropy.mean()

                loss = loss_pi + 0.5 * loss_v - 0.01 * loss_entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                kl = (logp_old_buf[batch_idx] - logp).mean().item()
                if kl > 1.5 * self.target_kl:
                    return  # Early stopping

    def train(self, epochs=50):
        obs, _ = self.env.reset()
        episode_lengths = []
        current_ep_len = 0

        for epoch in range(epochs):
            obs_buf, act_buf, adv_buf, ret_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], [], []
            for _ in range(self.steps_per_epoch):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action, logp, value = self.actor_critic.step(obs_tensor)
                clipped_action = torch.clamp(action, -self.act_limit, self.act_limit)
                next_obs, reward, terminated, truncated, _ = self.env.step(clipped_action.numpy().squeeze())
                done = terminated or truncated

                obs_buf.append(obs)
                act_buf.append(action.numpy().squeeze())
                logp_buf.append(logp.item())
                rew_buf.append(reward)
                val_buf.append(value.item())
                done_buf.append(done)

                current_ep_len += 1
                obs = next_obs
                if done:
                    episode_lengths.append(current_ep_len)  # ⬅ Save episode length
                    current_ep_len = 0
                    obs, _ = self.env.reset()

            val_buf.append(0)
            adv_buf = self.compute_advantages(rew_buf, val_buf, done_buf)
            ret_buf = adv_buf + val_buf[:-1]

            self.update(obs_buf, act_buf, adv_buf, ret_buf, logp_buf)
            self.reward_avg.append(np.mean(rew_buf))

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp, value = self.actor_critic.step(state_tensor)

        if not evaluate:
            clipped_action = torch.clamp(action, -self.act_limit, self.act_limit)
            return clipped_action.cpu().numpy().squeeze(), logp, value
        return action.cpu().numpy().squeeze()

    # def record_video(self, video_path="ppo_bipedal_videos", iter = 0, episode_length=1600):
    #     env = RecordVideo(
    #         self.env,
    #         video_folder=video_path,
    #         episode_trigger=lambda ep: True,
    #         name_prefix=f"ppo_bipedal_iter_{iter}"
    #     )
    #     obs, _ = env.reset()
    #     total_reward = 0
    #     for _ in range(episode_length):
    #         obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    #         action, _, _ = self.actor_critic.step(obs_tensor)
    #         action_clipped = torch.clamp(action, -self.act_limit, self.act_limit)
    #         obs, reward, terminated, truncated, _ = env.step(action_clipped.numpy().squeeze())
    #         total_reward += reward
    #         if terminated or truncated:
    #             break
    #     env.close()
    #     print(f"Episode finished. Total reward: {total_reward}")
    #     print(f"Video saved to: {os.path.abspath(video_path)}")
    #
    # def plot_reward_avg(self):
    #     plt.plot(self.reward_avg)
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Average Reward")
    #     plt.title("Average Reward per Epoch")
    #     plt.show()