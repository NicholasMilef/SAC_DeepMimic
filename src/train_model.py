import os
import pdb
import argparse

import gym
from Solvers.AbstractSolver import AbstractSolver
from Solvers.SAC import SAC
from Solvers.PPO import PPO
from Solvers.REINFORCE import REINFORCE
from pybullet_envs.deep_mimic.gym_env import HumanoidDeepMimicWalkBulletEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', help="RL method to use")
    parser.add_argument('-r', '--replay', help="replay buffer prioritization scheme", default='uniform')
    parser.add_argument('-w', '--warmup', help="warm-up transitions", type=int, default=5000)
    args = parser.parse_args()

    num_iterations = 100000
    #env = gym.make("HumanoidDeepMimicWalkBulletEnv-v1")
    env = gym.make("HumanoidBulletEnv-v0")

    options = {
        'lr': 0.001,
        'gamma': 0.95,
        'replay_memory_size': 1000000,
        'batch_size': 128,
        'replay': args.replay,
        'warmup': args.warmup
    }

    if args.method == 'sac':
        solver = SAC(env, options)
    elif args.method == 'r':
        solver = REINFORCE(env, options)
    elif args.method == 'ppo':
        solver = PPO(env, options)
    else:
        print('Unsupported Method')
        exit()

    env.render(mode="human")
    for i in range(num_iterations):

        solver.train_episode(i)

    env.close()
    print("Finished training")


if __name__ == "__main__":
    main()
