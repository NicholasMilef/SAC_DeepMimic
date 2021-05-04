import os
import pdb
import math
import numpy as np
from Solvers.AbstractSolver import AbstractSolver
import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# We reimplemented REINFORCE from our class assignment in PyTorch
# We modified it to work with continous actions
def LossREINFORCE(output, labels, G):
	loss = 0
	x = torch.clip(output, 1e-3, 1)
	mu = x[:,:,0]
	sigma = x[:,:,1]

	loss = torch.clip((1.0 / (torch.pow(sigma,2)*math.sqrt(2*math.pi))) * torch.exp(-0.5 * torch.pow((labels - mu) / torch.pow(sigma,2), 2)), 1e-3, 1)
	loss = -torch.mean(G*torch.log(loss))
	return loss

# REINFORCE policy using neural network
class ModelREINFORCE(nn.Module):
	def __init__(self, state_size, action_size):
		super(ModelREINFORCE, self).__init__()
		self.fc0 = nn.Linear(state_size[0], 1024)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, action_size*2)

	def forward(self, x):
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		x = F.relu(torch.reshape(x, (x.shape[0], x.shape[1]//2, 2)))
		#x[:,:,1] = F.relu(x[:,:,1]) # sigma can't be negative
		return x

# REINFORCE training class
class REINFORCE(AbstractSolver):
	def __init__(self, env, options):
		super().__init__(env, options)
		self.state_size = (self.env.observation_space.shape[0],)
		self.action_size = len(self.env.action_space.sample())

		# create model
		self.model = ModelREINFORCE(self.state_size, self.action_size)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options['lr'])

		self.policy = self.create_greedy_policy()
		self.trajectory = []

	# START CODE ADAPTED FROM HW ASSIGNMENT
	def create_greedy_policy(self):
		def policy_fn(state):
			output = None
			with torch.no_grad():
				data = torch.from_numpy(state.reshape(1, state.shape[0])).float()
				output = self.model(data)
			return output
		return policy_fn
	# END CODE ADAPTED FROM HW ASSIGNMENT

	@staticmethod
	def chooseAction(p):
		p = p.numpy()
		mu = p[:,:,0]
		sigma = p[:,:,1]

		a = np.random.normal(mu, sigma).flatten()
		return a

	@staticmethod
	def scaleAction(a, env):
		a_range = env.action_space.high - env.action_space.low
		a = env.action_space.low + (a_range * a)
		return a

	def train_episode(self, iteration):
		# initialize episode
		s = self.env.reset()

		self.model.zero_grad()

		# generate episode
		done = False
		while not done:
			a_prob = self.policy(s)
			a = REINFORCE.chooseAction(a_prob)
			s_p, r, done, _ = self.env.step(REINFORCE.scaleAction(a, self.env))
			self.trajectory.append((s, s_p, a, r))
			s = s_p
		
		# train network
		Gs = np.zeros((len(self.trajectory),1))
		G = 0
		states = np.zeros((len(self.trajectory), self.state_size[0]))
		labels = np.zeros((len(self.trajectory), self.action_size))
		for i in range(len(self.trajectory)-1, -1, -1):
			s, s_p, a, r = self.trajectory[i]

			G = r + self.options['gamma'] * G
			Gs[i] = G

			states[i] = s
			labels[i] = a

		output = self.model(torch.from_numpy(states).float())
		labels = torch.from_numpy(labels).detach().float()
		G = torch.tensor(G, requires_grad=False)

		# convert to tensors
		loss = LossREINFORCE(output, labels, G)
		loss.backward()

		history = {
			'loss': float(loss),
			'episode_length': len(self.trajectory),
			'return': float(G)
		}

		self.plot_info(history, iteration, 10)
		torch.save(self.model.state_dict(), 'REINFORCE.pth')

		self.trajectory.clear()

	def print_name(self):
		return 'REINFORCE'