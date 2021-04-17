import os
import abc

class AbstractSolver(abc.ABC):
	def __init__(self, env, options):
		self.env = env

	def step(self, action):
		pass

	def run_greedy():
		pass

	@abc.abstractmethod
	def train_episode(self):
		pass