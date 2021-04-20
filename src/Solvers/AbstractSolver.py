import os
import abc
import pdb
import datetime
from torch.utils.tensorboard import SummaryWriter

class AbstractSolver(abc.ABC):
	def __init__(self, env, options):
		self.options = options
		self.env = env
		self.cumulative_loss = 0
		self.cumulative_episode_length = 0
		self.total_return = 0
		self.writer = SummaryWriter(comment=self.print_name())

	def step(self, action):
		pass

	def run_greedy():
		pass

	# Plots training information in Tensorboard
	# history: history object
	# iteration: current iteration number
	# frequency: number of iterations before plotting number
	def plot_info(self, history, iteration, frequency):
		loss = history['loss']
		episode_length = history['episode_length']
		total_return = history['return']
		if (iteration + 1) % frequency == 0:		
			self.writer.add_scalar('loss', loss, iteration)
			self.writer.add_scalar('episode_length', episode_length, iteration)
			self.writer.add_scalar('total_return', total_return, iteration)
			self.cumulative_loss /= frequency
			self.cumulative_episode_length /= frequency
			self.total_return /= frequency

			print('Loss: %.2f' % self.cumulative_loss,
				'\tEpisode Length:', self.cumulative_episode_length,
				'\tTotal Return: %.2f' % self.total_return)

			self.cumulative_loss = 0
			self.cumulative_episode_length = 0
			self.total_return = 0
		else:
			self.cumulative_loss += loss
			self.cumulative_episode_length += episode_length
			self.total_return += total_return

	@abc.abstractmethod
	def train_episode(self, iteration):
		pass

	@abc.abstractmethod
	def print_name(self):
		pass

