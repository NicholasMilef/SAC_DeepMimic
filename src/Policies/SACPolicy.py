import os
import torch
import numpy as np
import pdb
from Policies.AbstractPolicy import AbstractPolicy
from Solvers.SAC import PolicySAC


class SACPolicy(AbstractPolicy):
    def __init__(self, env):
        super(SACPolicy, self).__init__()

        self.env = env
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = len(self.env.action_space.sample())

        self.model = None

    def loadPolicy(self, name):
        self.model = PolicySAC(self.state_size, self.action_size)
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def run(self, states):
        actions = self.model.get_action(states)
        return actions
