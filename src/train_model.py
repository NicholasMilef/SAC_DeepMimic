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

    num_iterations = 100000
    env = gym.make("HumanoidDeepMimicWalkBulletEnv-v1")

    options = {
        'lr': 0.001,
        'gamma': 0.95
    }

    if args.method == 'sac':
        solver = SAC(env, options)
    elif args.method == 'r':
        solver = REINFORCE(env, options)
    else:
        print('Unsupported Method')
        exit()

    for i in range(num_iterations):
        env.render(mode="human")
        solver.train_episode()

    env.close()
    print("Finished training")


if __name__ == "__main__":
    main()
