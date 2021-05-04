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
import pybullet_envs
import queue
import pdb


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
        loc = torch.zeros(mean.size())
        scale = torch.ones(mean.size())
        if mean.size()[1] == 1:
            normal = Normal(loc, scale)
            z = normal.sample()

        else:
            scale = torch.diag_embed(scale)
            mvn = MultivariateNormal(loc, scale)
            z = mvn.sample()

        action = torch.tanh(mean + std * z)
        squashing = torch.log(1 - action.pow(2) + epsilon).sum(1)
        if mean.size()[1] == 1:
            log_prob = torch.reshape(Normal(mean, std).log_prob(mean + std * z), (-1,)) - squashing

        else:
            scale_std = torch.diag_embed(std)
            log_prob = MultivariateNormal(mean, scale_std).log_prob(mean + std * z) - squashing

        log_prob = torch.reshape(log_prob, (-1, 1))

        #log_prob = log_prob.sum(1, keepdim=True)
        # print(torch.diag(std).size())
        #log_prob = MultivariateNormal(mean, torch.diag(std)).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        #log_prob = log_prob.sum(1, keepdim=True)
        # log_prob = MultivariateNormal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        # z = mvn.sample()
        loc = torch.zeros(mean.size())
        scale = torch.ones(mean.size())
        if mean.size()[1] == 1:
            normal = Normal(loc, scale)
            z = normal.sample()
        else:
            scale = torch.diag_embed(scale)
            mvn = MultivariateNormal(loc, scale)
            z = mvn.sample()

        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, value=0):
        value = 0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, np.full_like(done, 1), np.full_like(done, -1)

    def __len__(self):
        return len(self.buffer)

class PER(ReplayBuffer):
    def __init__(self, capacity, state_size, action_size):
        super(PER, self).__init__(capacity)
        self.epsilon = 0.001
        self.buffer = []
        self.state_size = state_size
        self.action_size = action_size
        self.age = 0

        self.alpha = 1.0#1.0
        self.beta = 0.0

        self.sum = 0

    def updateSum(self, value, oldValue):
        self.sum = self.sum - oldValue + value

    def push(self, state, actions, reward, next_state, done, value=0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        index = len(self.buffer)-1

        oldValue = 0
        if len(self.buffer) == self.capacity:
            index == 0
            for b in self.buffer:
                if b[2] == self.position:                    
                    oldValue, _, _ = self.buffer[index]
                    break
                index += 1

        self.updateSum(oldValue, value)

        self.buffer[index] = (value, (state, actions, reward, next_state, done), self.position)
        self.position = (self.position + 1) % self.capacity
        self.buffer.sort(key=lambda t: t[0], reverse=True)
        return 

    def sample(self, batch_size):
        state = np.zeros((batch_size, self.state_size[0]))
        action = np.zeros((batch_size, self.action_size))
        reward = np.zeros((batch_size))
        next_state = np.zeros((batch_size, self.state_size[0]))
        done = np.zeros((batch_size))
        weight = np.zeros((batch_size))
        indices = np.zeros((batch_size))

        for i in range(batch_size):
            s, a, r, n_s, d, w, index = self.sample_one()
            state[i] = s
            action[i] = a
            reward[i] = r
            next_state[i] = n_s
            done[i] = d
            weight[i] = w
            indices[i] = index

        return state, action, reward, next_state, done, weight, indices

    def p(self, i):
        return (1.0 / float(i+1))**self.alpha

    def prob(self, i):
        return (self.p(i)**self.alpha) / self.sum

    def sample_one(self):
        number = random.uniform(0, 1)
        i = 0
        cI = 0

        while i < len(self.buffer):
            if number <= self.prob(i):
                cI = 0
                break
            i += 1

        value, (state, action, reward, next_state, done), index = self.buffer[cI]
        weight = ((1.0 / len(self.buffer)) * (1.0 / self.prob(cI)))**self.beta
        max_weight = 1.0 / len(self.buffer)
        weight = 1.0
        max_weight = 1.0
        return state, action, reward, next_state, done, weight / max_weight, cI

    def update(self, indices, values):
        for i in range(len(indices)):
            value, data, index = self.buffer[int(indices[i])]
            self.buffer[int(indices[i])] = (values[i], data, index)

    def __len__(self):
        return len(self.buffer)

class SAC(AbstractSolver):
    def __init__(self, env, options):
        super(SAC, self).__init__(env, options)

        #self.env = NormalizedActions(gym.make("Pendulum-v0"))
        #self.env = NormalizedActions(gym.make("HumanoidBulletEnv-v0"))
        self.env = NormalizedActions(env)
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = len(self.env.action_space.sample())

        if options['replay'] == 'uniform':
            self.replay_buffer = ReplayBuffer(self.options['replay_memory_size'])
        elif options['replay'] == 'per':
            self.replay_buffer = PER(self.options['replay_memory_size'], self.state_size, self.action_size)

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

    def getTDError(self, state, reward, next_state, gamma):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(np.array([reward])).unsqueeze(1)

        
        with torch.no_grad():
            self.VF.eval()
            td_error = torch.abs(reward + gamma * self.VF(next_state) - self.VF(state))
            self.VF.train()

        return td_error

    def update(self, batch_size, gamma=0.99):

        state, action, reward, next_state, done, weight, index = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)
        weight = torch.FloatTensor(np.float32(weight)).unsqueeze(1)

        # Update priority with TD error
        if self.options['replay'] == 'per':

            with torch.no_grad():
                self.VF.eval()
                td_error = torch.abs(reward + gamma * self.VF(next_state) - self.VF(state))
                self.replay_buffer.update(index, td_error)
                self.VF.train()

        predicted_q_value1 = self.QF1(state, action)
        predicted_q_value2 = self.QF2(state, action)
        predicted_value = self.VF(state)
        new_action, log_prob, epsilon, mean, log_std = self.actor.evaluate(state)

        # Training Q Function
        target_value = self.target_VF(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = nn.MSELoss()(predicted_q_value1*weight, target_q_value.detach()*weight)
        q_value_loss2 = nn.MSELoss()(predicted_q_value2*weight, target_q_value.detach()*weight)

        self.QF1_optimizer.zero_grad()
        q_value_loss1.backward()
        self.QF1_optimizer.step()
        self.QF2_optimizer.zero_grad()
        q_value_loss2.backward()
        self.QF2_optimizer.step()

        # Training Value Function
        predicted_new_q_value = torch.min(self.QF1(state, new_action), self.QF2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = nn.MSELoss()(predicted_value*weight, target_value_func.detach()*weight)

        self.VF_optimizer.zero_grad()
        value_loss.backward()
        self.VF_optimizer.step()

        # Training Policy Function
        policy_loss = ((log_prob - predicted_new_q_value)*weight).mean()

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
        state = self.env.reset()
        accum_rewards = 0
        done = False
        st = 1

        while not done:
            if len(self.replay_buffer) > self.options['warmup']:
                action = self.actor.get_action(state).detach()
                next_state, reward, done, _ = self.env.step(action.numpy())
            else:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action[0])

            gamma = 0.99
            #td_error = self.getTDError(state, reward, next_state, gamma)
            td_error = 1000
            self.replay_buffer.push(state, action, reward, next_state, done, td_error)

            state = next_state
            accum_rewards += reward
            st += 1

            if len(self.replay_buffer) > self.options['batch_size']:
                self.update(self.options['batch_size'], self.options['gamma'])

        # record data
        history = {
            'loss': 0,#float(loss),
            'episode_length': st,
            'return': float(accum_rewards)
        }

        self.plot_info(history, iteration, 10)

        if iteration % 5000 == 0:
            torch.save(self.actor.state_dict(),
                'SAC_epi_' + str(iteration) +
                '_' + self.options['replay'] +
                '_t' + str(self.options['warmup']) + '.pth')
        print(iteration, accum_rewards, st)

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
