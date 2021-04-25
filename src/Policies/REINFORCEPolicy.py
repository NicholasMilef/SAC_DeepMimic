import os
import torch
import numpy as np
import pdb
from Policies.AbstractPolicy import AbstractPolicy
from Solvers.REINFORCE import ModelREINFORCE, REINFORCE

class REINFORCEPolicy(AbstractPolicy):
    def __init__(self, env):
        super(REINFORCEPolicy, self).__init__()

        self.env = env
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = len(self.env.action_space.sample())

        self.model = None

    def loadPolicy(self, name):
        self.model = ModelREINFORCE(self.state_size, self.action_size)
        self.model.load_state_dict(torch.load(name))
        self.model.eval()

    def run(self, states):
        s = np.expand_dims(states, axis=0)
        action_probs = self.model(torch.from_numpy(s).float()).detach()
        actions = REINFORCE.scaleAction(REINFORCE.chooseAction(action_probs), self.env)
        return actions