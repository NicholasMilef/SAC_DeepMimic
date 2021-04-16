import os
import pdb

import gym
import pybulletgym
from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicWalkBulletEnv

def train_episode(env):
    s = env.reset()

    done = False
    while True:#while not done:
        s_p, r, done, _ = env.step(env.action_space.sample())
        s = s_p
        env.render()
        #pdb.set_trace()

def main():
    env = gym.make("HumanoidDeepMimicWalkBulletEnv-v1")

    num_iterations = 100

    for i in range(num_iterations):
        env.render(mode="human")
        train_episode(env)

    env.close()

if __name__ == "__main__":
    main()