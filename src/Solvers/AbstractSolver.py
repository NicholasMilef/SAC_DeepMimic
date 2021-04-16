import os

class AbstractSolver:
	def __init__(self, env, options):
		self.env = env

	def step(self, action):
		pass

	def run_greedy():
		pass

	@abstractmethod
	def train_episode(self):
		pass