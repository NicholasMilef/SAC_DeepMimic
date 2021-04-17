import os
import pdb
import keras
from keras.layers import Input, Dense, Softmax
from keras.optimizers import Adam
from keras.models import Model
from Solvers.AbstractSolver import AbstractSolver

def pg_loss(rewards):
	def loss(labels, predicted_output):
		return 0
	return loss

keras.losses.pg_loss = pg_loss

class REINFORCE(AbstractSolver):
	def __init__(self, env, options):
		super().__init__(env, options)
		self.state_size = (self.env.observation_space.shape[0],)
		self.action_size = len(self.env.action_space.sample())
		self.policy = self.build_model()

	# Code adapted from HW assignment
	def build_model(self):
		rewards = Input(shape=(1,))
		layers = 3
		states = Input(shape=(self.state_size))
		d = states
		for l in range(layers):
			d = Dense(l, activation='relu')(d)
		do = Dense(self.action_size)(d)
		out = Softmax()(do)

		opt = Adam(lr=self.options['lr'])
		model = Model(inputs=[states, rewards], outputs=out)
		model.compile(optimizer=opt, loss=pg_loss(rewards))
		return model

	def train_episode(self):
		s = self.env.reset()

		done = False
		while not done:
			s_p, r, done, _ = self.env.step(self.env.action_space.sample())
			s = s_p		
