# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Softmax, Input, Lambda
from keras.optimizers import Adam
from keras.losses import MSE
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from skimage.transform import resize
from skimage import color
from collections import deque
import tensorflow as tf
# import tensorflow_probability as tfp
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class SAC(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = (1,)
        self.trajectory = []
        self.replay_buffer = deque()
        # To escape from the maximum bias, we need two independent Q functions.
        self.QF1 = self.build_soft_Q()
        self.QF2 = self.build_soft_Q()

        # They set two V functions for stability.
        self.VF = self.build_soft_V()
        self.target_VF = self.build_soft_V()

        # self.initiate_target_VF()  # start from the same network.

        self.actor = self.build_actor()  # distribution(network) of policy

        self.policy = self.create_greedy_policy()  # policy maker
        # temperature
        self.alpha = 0.2
        # moving average parameter
        self.tau = 0.01

    def build_soft_Q(self):
        # A soft Q function.
        # input : [states, actions]
        # output : values
        # loss : mean squared error(MSE).
        # The counterpart value : r(s,a) + gamma * self.target_VF.predict(next_s) from batch in the replay buffer.
        layers = self.options.layers
        states = Input(shape=self.state_size)
        actions = Input(shape=self.action_size)
        d = keras.layers.Concatenate(axis=1)([states, actions])
        for l in layers:
            d = Dense(l, activation='relu')(d)
        values = Dense(1, activation='linear', name='Q_value')(d)
        model = Model(inputs=[states, actions], outputs=values)
        model.compile(optimizer=Adam(lr=self.options.alpha), loss=MSE)
        return model

    def build_soft_V(self):
        # A soft V function.
        # input : [states]
        # output : values
        # loss : mean squared error(MSE).
        # The counterpart value : Q(s,a) - alpha * log (self.actor.predict(s,a))
        # a ~ self.policy(s) : the action is sampled from the current policy(on policy)
        layers = self.options.layers
        states = Input(shape=self.state_size)
        d = states
        for l in layers:
            d = Dense(l, activation='relu')(d)
        values = Dense(1, activation='linear', name='state_value')(d)
        model = Model(inputs=states, outputs=values)
        model.compile(optimizer=Adam(lr=self.options.alpha), loss=MSE)
        return model

    def initiate_target_VF(self):
        # copy weights from model to target_model
        self.target_VF.set_weights(self.VF.get_weights())

    def update_target_VF(self):
        """Create tensorflow operations for updating target value function."""
        x = self.target_VF.get_weights()
        for i in range(len(x)):
            x[i] = (1 - self.tau) * self.target_VF.get_weights()[i] + self.tau * self.VF.get_weights()[i]
        self.target_VF.set_weights(x)

    def build_actor(self):
        # A squashed Gaussian policy.
        # input : [states, actions]
        # output : [values, log_liks]
        # In our case, each value should be bounded in [0,1] -> squashing the layer with sigmoid ftn.
        layers = self.options.layers
        states = Input(shape=self.state_size)
        d = states
        for l in layers:
            d = Dense(l, activation='relu')(d)
        means = Dense(1, activation='linear', name='mean')(d)
        log_std = Dense(1, activation='linear', name='log_std')(d)

        def sampling(args):
            means, log_std = args
            epsilon = K.random_normal(shape=K.shape(means))
            return means + K.exp(log_std) * epsilon

        raw_actions = Lambda(sampling, output_shape=(1,))([means, log_std])
        actions = Dense(1, activation='sigmoid', name='actions')(raw_actions)

        model = Model(inputs=states, outputs=[actions, raw_actions, means, log_std])
        model.compile(optimizer=Adam(lr=self.options.alpha), loss=self.actor_loss(states))
        return model
        # self.actor_loss(self.QF1, self.QF2, states, actions, raw_actions, means, log_std))

    def actor_loss(self, states):
        def loss(y_true, y_pred):

            actions = y_pred[0]
            raw_actions = y_pred[1]
            means = y_pred[2]
            log_std = y_pred[3]

            log_std = K.clip(log_std, -20., 2.)
            log_liks = -1 / 2 * log_std - 1 / 2 * ((raw_actions - means) / K.exp(log_std))**2
            shape = K.int_shape(states)
            if shape[0] is not None:
                Q_value = tf.minimum(
                    tf.squeeze(self.QF1.predict([states, actions]), axis=1),
                    tf.squeeze(self.QF2.predict([states, actions]), axis=1)
                )
            else:
                Q_value = np.zeros_like(log_liks)
            return tf.reduce_mean(0.2 * log_liks - Q_value)
        return loss

    def create_greedy_policy(self):

        # Indeed, this problem is a discrete case, the predicted value is in [0,1]
        # I am not sure if we need to do 0.5 cut-off for this case.(on policy)
        # Or np.random.choice(np.arange(len(action_probs)), p=value)

        def policy_fn(state):

            return 1 * (self.actor.predict([[state]])[0][0][0] > 0.5)  # first option.

        return policy_fn

    def discounted_reward(self, rewards):

        discounted_rewards, cumulative_reward = np.zeros_like(rewards), 0
        for t in reversed(range(0, len(rewards))):
            cumulative_reward = rewards[t] + cumulative_reward * self.options.gamma
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards

    def train_episode(self):

        state = self.env.reset()
        # state = np.reshape(state, [1, self.state_size[0]])
        done = False
        t = 1
        while not done and t < self.options.steps:
            action = self.policy(state)
            next_state, reward, done, _ = self.step(action)
            if len(self.replay_buffer) == self.options.replay_memory_size:
                self.replay_buffer.popleft()
            self.replay_buffer.append([state, action, reward, next_state, done])
            if len(self.replay_buffer) < self.options.batch_size:
                t += 1
                state = next_state
                continue
            # sample the batch_index
            batch_index = np.random.choice(len(self.replay_buffer), self.options.batch_size, replace=False)
            states = np.zeros((self.options.batch_size, self.state_size[0]))
            actions = np.zeros((self.options.batch_size, ))
            rewards = np.zeros((self.options.batch_size, ))
            next_states = np.zeros((self.options.batch_size, self.state_size[0]))
            not_dones = np.zeros((self.options.batch_size, ))
            for i, idx in enumerate(batch_index):
                s, a, r, n, d = self.replay_buffer[idx]
                states[i] = s
                actions[i] = a
                rewards[i] = r
                next_states[i] = n
                not_dones[i] = 1 - 1 * d
            # update soft V function : need to think QF1, QF2.
            Q_value = np.minimum(
                np.squeeze(self.QF1.predict([states, actions]), axis=1),
                np.squeeze(self.QF2.predict([states, actions]), axis=1)
            )
            _, raw_actions, means, log_std = self.actor.predict(states)
            log_liks = -1 / 2 * log_std - 1 / 2 * ((raw_actions - means) / np.exp(log_std))**2
            y = Q_value - self.alpha * np.squeeze(log_liks, axis=1)
            self.VF.fit(states, y, verbose=0)
            # update soft Q functions : QF1, QF2
            z = rewards + self.options.gamma * not_dones * np.squeeze(self.target_VF.predict(next_states), axis=1)
            self.QF1.fit([states, actions], z, verbose=0)
            self.QF2.fit([states, actions], z, verbose=0)

            # update actor function.
            # self.actor.fit(states, [np.zeros_like(_), np.zeros_like(raw_actions), np.zeros_like(means), np.zeros_like(log_std)], verbose=1)  # the second argument is not necessary.
            # moving average of parameters of self.target_VF and self.VF.
            self.update_target_VF()
            t += 1
        ################################

    def __str__(self):
        return "SAC"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
