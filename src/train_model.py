import os
import pdb
import argparse

import gym
from Solvers.AbstractSolver import AbstractSolver
from Solvers.SAC import SAC
from Solvers.REINFORCE import REINFORCE
from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicWalkBulletEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', help="RL method to use")
    args = parser.parse_args()

    if args.method == 'sac':
        solver = SAC()
    elif args.method == 'r':
        solver = REINFORCE()
    else:
        print('Unsupported Method')
        exit()

    env = gym.make("HumanoidDeepMimicWalkBulletEnv-v1")

    num_iterations = 100

    for i in range(num_iterations):
        env.render(mode="human")
        solver.train_episode(env)

    env.close()


if __name__ == "__main__":
    main()
