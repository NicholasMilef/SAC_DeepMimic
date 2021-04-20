import gym
import numpy as np
from collections import deque

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from Solvers.AbstractSolver import AbstractSolver


class ValueFunction(nn.Module):
    def __init__(self, state_size, init_w=3e-3):
        super(ValueFunction, self).__init__()

        self.fc0 = nn.Linear(state_size[0], 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)

        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc0(state))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SoftQFunction(nn.Module):
    def __init__(self, state_size, action_size, init_w=3e-3):
        super(SoftQFunction, self).__init__()

        self.fc0 = nn.Linear(state_size[0] + action_size, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)

        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PolicySAC(nn.Module):
    def __init__(self, state_size, action_size, init_w=3e-3):
        super(PolicySAC, self).__init__()

        self.log_std_min = -20
        self.log_std_max = 2

        self.fc0 = nn.Linear(state_size[0], 1024)
        self.fc1 = nn.Linear(1024, 512)

        self.fc_mean = nn.Linear(512, action_size)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

        self.fc_log_std = nn.Linear(512, action_size)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc0(state))
        x = F.relu(self.fc1(x))

        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        loc = torch.zeros(len(mean))
        scale = torch.ones(len(mean))
        #mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        #z = mvn.sample()
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        #log_prob = MultivariateNormal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        #mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        #z = mvn.sample()
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


class SAC(AbstractSolver):
    def __init__(self, env, options):
        super().__init__(env, options)

        self.env = NormalizedActions(gym.make("Pendulum-v0"))
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = len(self.env.action_space.sample())
        self.replay_buffer = deque()
        # To escape from the maximum bias, we need two independent Q functions.
        self.QF1 = SoftQFunction(self.state_size, self.action_size)
        self.QF2 = SoftQFunction(self.state_size, self.action_size)

        # They set two V functions for stability.
        self.VF = ValueFunction(self.state_size)
        self.target_VF = ValueFunction(self.state_size)
        self.initiate_target_VF()  # start from the same network.
        self.actor = PolicySAC(self.state_size, self.action_size)  # distribution(network) of policy
        self.policy = self.create_greedy_policy()  # policy maker

        self.VF_optimizer = optim.Adam(self.VF.parameters(), lr=self.options['lr'])
        self.QF1_optimizer = optim.Adam(self.QF1.parameters(), lr=self.options['lr'])
        self.QF2_optimizer = optim.Adam(self.QF2.parameters(), lr=self.options['lr'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.options['lr'])

        # temperature
        self.alpha = 0.2
        # moving average parameter
        self.tau = 0.01

    def initiate_target_VF(self):
        for target_param, param in zip(self.target_VF.parameters(), self.VF.parameters()):
            target_param.data.copy_(param.data)

    def create_greedy_policy(self):
        def policy_fn(state):
            return self.actor.get_action(state).detach().cpu().numpy()
        return policy_fn

    def train_episode(self, iteration):
        t = 1
        while t < 40000:
            state = self.env.reset()
            accum_rewards = 0
            done = False

            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                if len(self.replay_buffer) == self.options['replay_memory_size']:
                    self.replay_buffer.popleft()
                self.replay_buffer.append([state, action, reward, next_state, done])
                if len(self.replay_buffer) < self.options['batch_size']:
                    state = next_state
                    continue
                # sample the batch_index
                batch_index = np.random.choice(len(self.replay_buffer), self.options['batch_size'], replace=False)
                states = np.zeros((self.options['batch_size'], self.state_size[0]))
                actions = np.zeros((self.options['batch_size'], ))
                rewards = np.zeros((self.options['batch_size'], ))
                next_states = np.zeros((self.options['batch_size'], self.state_size[0]))
                not_dones = np.zeros((self.options['batch_size'], ))
                for i, idx in enumerate(batch_index):
                    s, a, r, n, d = self.replay_buffer[idx]
                    states[i] = s
                    actions[i] = a
                    rewards[i] = r
                    next_states[i] = n
                    not_dones[i] = 1 - 1 * d

                states = torch.FloatTensor(states)
                next_states = torch.FloatTensor(next_states)
                actions = torch.FloatTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                not_dones = torch.FloatTensor(np.float32(not_dones)).unsqueeze(1)

                # Value function update
                new_action, log_prob, epsilon, mean, log_std = self.actor.evaluate(states)
                Q_value = torch.min(self.QF1(states, new_action), self.QF2(states, new_action))
                y = Q_value - self.alpha * log_prob
                value_loss = nn.MSELoss()(self.VF(states), y.detach())

                self.VF_optimizer.zero_grad()
                value_loss.backward()
                self.VF_optimizer.step()
                # Q functions update
                target_value_next = self.target_VF(next_states)
                z = rewards + self.options['gamma'] * not_dones * target_value_next

                QF1_loss = nn.MSELoss()(self.QF1(states, actions), z.detach())
                QF2_loss = nn.MSELoss()(self.QF2(states, actions), z.detach())

                self.QF1_optimizer.zero_grad()
                QF1_loss.backward()
                self.QF1_optimizer.step()
                self.QF2_optimizer.zero_grad()
                QF2_loss.backward()
                self.QF2_optimizer.step()

                # actor update
                actor_loss = (self.alpha * log_prob - Q_value.detach()).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                accum_rewards += reward
                state = next_state
                t += 1

            print(accum_rewards)

    def print_name(self):
        return 'SAC'


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

        return actions
