import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Any
import gymnasium as gym

# Assuming ActorCritic is defined elsewhere in the file
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        super(ActorCritic, self).__init__()
        self.device = device

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        ).to(device) # Ensure the actor network is on the device

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        ).to(device) # Ensure the critic network is on the device

        self.log_std = nn.Parameter(torch.zeros(action_dim)).to(device) # Ensure log_std is on the device


    def forward(self, state: torch.Tensor):
        # Ensure the input tensor is on the same device as the model parameters
        state = state.to(self.device)

        # Actor computes mean and standard deviation
        action_mean = self.actor(state)
        log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(log_std)
        dist = Normal(action_mean, action_std)

        # Critic computes value
        value = self.critic(state)

        return dist, value

    # Fix: Ensure the input tensor is on the same device as the model
    def step(self, obs: torch.Tensor):
        with torch.no_grad():
            # Move the input observation to the same device as the model
            # This line was already present, but the error indicates the model itself
            # wasn't fully on the device. Moving the sub-modules in __init__ fixes this.
            # obs = obs.to(next(self.parameters()).device) # This line might be redundant if the whole model is moved

            dist, value = self(obs) # Calls the forward method
            action = dist.sample()
            logp_a = dist.log_prob(action).sum(axis=-1)

        return action, value, logp_a

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        # Ensure the input tensors are on the same device
        state = state.to(self.device)
        action = action.to(self.device)

        dist, value = self(state)
        logp_a = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return value, logp_a, entropy


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, lr_actor: float, lr_critic: float, gamma: float, k_epochs: int, eps_clip: float, buffer_size: int, minibatch_size: int):
        self.device = device
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size

        self.actor_critic = ActorCritic(state_dim, action_dim, device).to(device) # Ensure the entire model is moved to the device
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': lr_actor},
            {'params': self.actor_critic.critic.parameters(), 'lr': lr_critic}
        ])

        self.buffer = RolloutBuffer(buffer_size)
        self.reward_avg = []

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        if not torch.is_tensor(state):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
             # Ensure the input tensor is on the correct device
             state_tensor = state.to(self.device)

        return self.actor_critic.step(state_tensor)

    def update(self):
        # Use the collected data
        old_states, old_actions, old_logps, old_values, advantages, returns = self.buffer.get()

        # Convert numpy arrays to tensors and move to device
        old_states = torch.tensor(old_states, dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(old_actions, dtype=torch.float32).to(self.device)
        old_logps = torch.tensor(old_logps, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Ensure advantages and returns are on the same device as model parameters
        advantages = advantages.to(next(self.actor_critic.parameters()).device)
        returns = returns.to(next(self.actor_critic.parameters()).device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy and value network
        for _ in range(self.k_epochs):
            # Create minibatches
            batch_size = old_states.size(0)
            for _ in range(batch_size // self.minibatch_size):
                indices = np.random.choice(batch_size, self.minibatch_size, replace=False)
                mini_states = old_states[indices]
                mini_actions = old_actions[indices]
                mini_old_logps = old_logps[indices]
                mini_advantages = advantages[indices]
                mini_returns = returns[indices]
                mini_old_values = old_values[indices]


                # Evaluate actions and values for the current minibatch
                values, logps, entropy = self.actor_critic.evaluate(mini_states, mini_actions)

                # Calculate policy loss
                ratios = torch.exp(logps - mini_old_logps.detach())
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mini_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                # Calculate value loss (clipped value loss)
                clipped_values = mini_old_values + torch.clamp(values - mini_old_values, -self.eps_clip, self.eps_clip)
                value_loss1 = (values - mini_returns)**2
                value_loss2 = (clipped_values - mini_returns)**2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()


                # Total loss
                loss = policy_loss + value_loss

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

class RolloutBuffer:
    def __init__(self, buffer_size: int):
        self.states = []
        self.actions = []
        self.logps = []
        self.values = []
        self.rewards = []
        self.is_terminals = []
        self.buffer_size = buffer_size

    def add(self, state: np.ndarray, action: np.ndarray, logp: float, value: float, reward: float, is_terminal: bool):
        self.states.append(state)
        self.actions.append(action)
        self.logps.append(logp)
        self.values.append(value)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def get(self):
        # Convert lists to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        logps = np.array(self.logps)
        values = np.array(self.values)
        rewards = np.array(self.rewards)
        is_terminals = np.array(self.is_terminals)

        # Calculate advantages and returns using Generalized Advantage Estimation (GAE)
        advantages = []
        returns = []
        last_gae_lam = 0
        gamma = 0.99 # Assume gamma is defined or passed to the buffer
        lam = 0.95 # Assume lambda is defined or passed to the buffer

        for t in reversed(range(len(rewards))):
            if t + 1 < len(rewards):
                next_non_terminal = 1.0 - is_terminals[t+1]
                next_value = values[t+1]
            else:
                next_non_terminal = 0.0
                next_value = 0.0

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)

        returns = advantages + values

        # Clear the buffer
        self.clear()

        return states, actions, logps, values, np.array(advantages), np.array(returns)


    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.values = []
        self.rewards = []
        self.is_terminals = []