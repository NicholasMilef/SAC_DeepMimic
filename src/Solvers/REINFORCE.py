import os
import pdb
import keras
import math
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Softmax
from keras.optimizers import Adam
from keras.models import Model
from Solvers.AbstractSolver import AbstractSolver
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def pg_loss(rewards):
	def loss(labels, predicted_output):
		l = 0
		G = rewards
		x = K.reshape(predicted_output, (K.shape(predicted_output)[0], K.shape(predicted_output)[1]//2, 2))
		x = K.clip(x, 1e-3, 1)
		mu = x[:,:,0]
		sigma = x[:,:,1]

		l = K.clip((1.0 / (K.pow(sigma,2)*math.sqrt(2*math.pi))) * K.exp(-0.5 * K.pow((labels - mu) / K.pow(sigma,2), 2)), 1e-3, 1)
		l = -K.mean(G*K.log(l))
		return l

	return loss

keras.losses.pg_loss = pg_loss

class REINFORCE(AbstractSolver):
	def __init__(self, env, options):
		super().__init__(env, options)
		self.state_size = (self.env.observation_space.shape[0],)
		self.action_size = len(self.env.action_space.sample())
		self.model = self.build_model()
		self.policy = self.create_greedy_policy()
		self.trajectory = []

	# START CODE ADAPTED FROM HW ASSIGNMENT
	def create_greedy_policy(self):
		def policy_fn(state):
			return self.model.predict([[state], np.zeros((1,1))])[0]
		return policy_fn
	
	def build_model(self):
		rewards = Input(shape=(1,))
		states = Input(shape=(self.state_size))
		d = states
		d = Dense(1024, activation='relu')(d)
		d = Dense(512, activation='relu')(d)
		do = Dense(self.action_size*2)(d) # outputting both mu and sigma for Gaussian
		out = Softmax()(do)

		opt = Adam(lr=self.options['lr'])
		model = Model(inputs=[states, rewards], outputs=out)
		model.compile(optimizer=opt, loss=pg_loss(rewards))
		return model

	# END CODE ADAPTED FROM HW ASSIGNMENT

	def chooseAction(self, p):
		x = np.reshape(p, (p.shape[0]//2, 2))
		mu = x[:,0]
		sigma = x[:,1]

		a = np.random.normal(mu, sigma).flatten()
		return a

	def scaleAction(self, a):
		a_range = self.env.action_space.high - self.env.action_space.low
		a = self.env.action_space.low + (a_range * a)
		return a

	def train_episode(self):
		# initialize episode
		s = self.env.reset()

		# generate episode
		done = False
		while not done:
			a_prob = self.policy(s)
			a = self.chooseAction(a_prob)
			s_p, r, done, _ = self.env.step(self.scaleAction(a))
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

		self.model.fit([states, Gs], labels, batch_size=len(self.trajectory), epochs=1)
		self.trajectory.clear()