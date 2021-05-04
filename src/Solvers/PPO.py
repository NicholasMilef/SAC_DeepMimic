import gym
import numpy as np
from collections import deque
import math
import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from Solvers.AbstractSolver import AbstractSolver

# PPO implementation from https://github.com/nikhilbarhate99/PPO-PyTorch
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_std_init = 0.6
        # this is only for continuous case.
        self.action_var = torch.full((action_size,), self.action_std_init**2)

        self.actor = nn.Sequential(
            nn.Linear(state_size[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_size,), new_action_std ** 2)

    def forward(self):
        raise NotImplementedError

    def get_action(self, states):
        mean = self.actor(states)
        cov = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(mean, cov)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate(self, states, actions):
        mean = self.actor(states)
        cov_vec = self.action_var.expand_as(mean)
        cov = torch.diag_embed(cov_vec)
        dist = MultivariateNormal(mean, cov)
        if self.action_size == 1:
            actions = actions.reshape(-1, self.action_size)
        log_prob = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic(states)

        return log_prob, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO(AbstractSolver):
    def __init__(self, env, options):
        super().__init__(env, options)
        # self.env = NormalizedActions(env)
        # self.env = NormalizedActions(gym.make("Pendulum-v0"))
        self.env = NormalizedActions(gym.make("BipedalWalker-v2"))
        # self.env = NormalizedActions(gym.make("HumanoidBulletEnv-v0"))
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = len(self.env.action_space.sample())
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.0003},
            {'params': self.policy.critic.parameters(), 'lr': 0.001}
        ])
        self.old_policy = ActorCritic(self.state_size, self.action_size)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # PPO specific parameters
        self.K_epoch = 40
        self.max_ep_len = 200
        self.eps_clip = 0.2              # clip parameter for PPO
        self.update_timestep = self.max_ep_len * 4
        self.action_std_decay_rate = 0.99
        self.min_action_std = 0.2
        self.action_std = 0.6
        self.action_std_decay_freq = 5

    def set_action_std(self, new_action_std):
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.options['gamma'] * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        for _ in range(self.K_epoch):

                # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
    # clear buffer
        self.buffer.clear()

    def train_episode(self, iteration):
        t = 1
        epi = 1
        while t < 4000000:
            state = self.env.reset()
            accum_rewards = 0
            done = False
            step = 1

            while not done or step < self.max_ep_len:

                with torch.no_grad():
                    state = torch.FloatTensor(state)
                    action, log_prob = self.old_policy.get_action(state)
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(log_prob)
                next_state, reward, done, _ = self.env.step(action.detach().numpy().flatten())
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                t += 1
                step += 1
                accum_rewards += reward
                if t % self.update_timestep == 0:
                    self.update()

                if t % self.action_std_decay_freq == 0:
                    self.action_std = max(self.action_std * self.action_std_decay_rate, self.min_action_std)
                    self.set_action_std(self.action_std)
            epi += 1
            if epi % 50 == 0:
                torch.save(self.policy.state_dict(), 'PPO_epi_{}.pt'.format(epi))
            print(epi, accum_rewards, step, t, self.action_std)

    def print_name(self):
        return "PPO"


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action
