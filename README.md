# DeepMimic reimplementation

A reimplementation of the DeepMimic paper with soft-actor critic. This code is based on Pytorch.

## dependency.

to be updated.


## Code Reference:

We referred and modified our code based on the following repositories.

1. PPO : https://github.com/nikhilbarhate99/PPO-PyTorch
2. SAC : https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
3. Deepmimic-environment : https://github.com/bulletphysics/bullet3/tree/e0b9bc6d7f0691bd88b758a7e94c7948bf315aa7/examples/pybullet/gym/pybullet_envs/deep_mimic/env


## Quick start.

1. Training model.

```
python src/train_model.py -m ['r', 'sac', 'ppo'] -r ['uniform', 'per']
```
For model selection, 'r' is for REINFORCE, 'sac' is for SAC, and 'ppo' is for PPO.

For replay buffer sampling selection, 'uniform' is uniform sampling, and 'per' is prioritized experience replay.

2. Testing model.

```
python src/testHumanoid.py -m ['r', 'sac', 'ppo'] -N 'pretrained_network_parameter.pth'
```
'pretrained_network_parameter.pth' should be alligned with the solver that you try to call.

