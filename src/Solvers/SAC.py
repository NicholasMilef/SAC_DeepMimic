import gym
import numpy as np
from collections import deque
import math
import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from Solvers.AbstractSolver import AbstractSolver
import pybullet_envs


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
        # loc = torch.zeros(len(mean))
        # scale = torch.ones(len(mean))
        # mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        # z = mvn.sample()
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # log_prob = MultivariateNormal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        # z = mvn.sample()
        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SAC(AbstractSolver):
    def __init__(self, env, options):
        super().__init__(env, options)

        #self.env = NormalizedActions(gym.make("HumanoidBulletEnv-v0"))
        self.env = NormalizedActions(env)
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = len(self.env.action_space.sample())
        self.replay_buffer = ReplayBuffer(self.options['replay_memory_size'])
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
        self.alpha = 1
        # moving average parameter
        self.tau = 0.01

    def initiate_target_VF(self):
        for target_param, param in zip(self.target_VF.parameters(), self.VF.parameters()):
            target_param.data.copy_(param.data)

    def update(self, batch_size, gamma=0.99):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        predicted_q_value1 = self.QF1(state, action)
        predicted_q_value2 = self.QF2(state, action)
        predicted_value = self.VF(state)
        new_action, log_prob, epsilon, mean, log_std = self.actor.evaluate(state)

    # Training Q Function
        target_value = self.target_VF(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = nn.MSELoss()(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = nn.MSELoss()(predicted_q_value2, target_q_value.detach())

        self.QF1_optimizer.zero_grad()
        q_value_loss1.backward()
        self.QF1_optimizer.step()
        self.QF2_optimizer.zero_grad()
        q_value_loss2.backward()
        self.QF2_optimizer.step()
    # Training Value Function
        predicted_new_q_value = torch.min(self.QF1(state, new_action), self.QF2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = nn.MSELoss()(predicted_value, target_value_func.detach())

        self.VF_optimizer.zero_grad()
        value_loss.backward()
        self.VF_optimizer.step()
    # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.target_VF.parameters(), self.VF.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def create_greedy_policy(self):
        def policy_fn(state):
            return self.actor.get_action(state).detach().cpu().numpy()
        return policy_fn

    def train_episode(self, iteration):
        t = 1
        epi = 1
        while t < 4000000:
            state = self.env.reset()
            accum_rewards = 0
            done = False
            st = 1

            while not done:

                if t > 5000:
                    action = self.actor.get_action(state).detach()
                    next_state, reward, done, _ = self.env.step(action.numpy())
                else:
                    action = self.env.action_space.sample()
                    next_state, reward, done, _ = self.env.step(action[0])

                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                accum_rewards += reward
                t += 1
                st += 1
                if len(self.replay_buffer) > self.options['batch_size']:
                    self.update(self.options['batch_size'])
            epi += 1
            if epi % 500 == 0:
                torch.save(self.actor.state_dict(), 'SAC_epi_{}.pt'.format(epi))
            print(epi, accum_rewards, st, t)

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

        return action
