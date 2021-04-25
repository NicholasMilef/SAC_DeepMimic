# Code adapted from:
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/env/testHumanoid.py

from pybullet_utils import bullet_client
import time
import math
import Viewer.motion_capture_data as motion_capture_data
from pybullet_envs.deep_mimic.env import humanoid_stable_pd
import pybullet_data
import pybullet as p1
import Viewer.humanoid_pose_interpolator as humanoid_pose_interpolator
import numpy as np

pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)

pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
z2y = pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
#planeId = pybullet_client.loadURDF("plane.urdf",[0,0,0],z2y)
planeId = pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                   z2y,
                                   useMaximalCoordinates=True)
pybullet_client.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)
#print("planeId=",planeId)

pybullet_client.configureDebugVisualizer(pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
pybullet_client.setGravity(0, -9.8, 0)

pybullet_client.setPhysicsEngineParameter(numSolverIterations=10)

mocapData = motion_capture_data.MotionCaptureData()
motionPath = pybullet_data.getDataPath()+"/data/motions/humanoid3d_walk.txt"
#motionPath = pybullet_data.getDataPath() + "/data/motions/humanoid3d_backflip.txt"
mocapData.Load(motionPath)
timeStep = 1. / 600
useFixedBase = False
humanoid = humanoid_stable_pd.HumanoidStablePD(pybullet_client, mocapData, timeStep, useFixedBase)
isInitialized = True

pybullet_client.setTimeStep(timeStep)
pybullet_client.setPhysicsEngineParameter(numSubSteps=2)
timeId = pybullet_client.addUserDebugParameter("time", 0, 10, 0)


def isKeyTriggered(keys, key):
  o = ord(key)
  if o in keys:
    return keys[ord(key)] & pybullet_client.KEY_WAS_TRIGGERED
  return False


# START OUR CODE: adding our policy
import pdb
from Policies.REINFORCEPolicy import REINFORCEPolicy
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', help='RL algorithm')
parser.add_argument('-N', '--name', help='name of model')
args = parser.parse_args()

env = gym.make("HumanoidDeepMimicWalkBulletEnv-v1")
policy = REINFORCEPolicy(env)
policy.loadPolicy(args.name)
# END OUR CODE

animating = False
singleStep = False
humanoid.resetPose()
t = 0
while (1):

  keys = pybullet_client.getKeyboardEvents()
  #print(keys)
  if isKeyTriggered(keys, ' '):
    animating = not animating

  if isKeyTriggered(keys, 'r'):
    humanoid.resetPose()

  if isKeyTriggered(keys, 'b'):
    singleStep = True

  if animating or singleStep:

    singleStep = False
    #t = pybullet_client.readUserDebugParameter(timeId)
    #print("t=",t)
    for i in range(1):

      #print("t=", t)
      humanoid.setSimTime(t)

      humanoid.computePose(humanoid._frameFraction)
      pose = humanoid._poseInterpolator
      #humanoid.initializePose(pose=pose, phys_model = humanoid._sim_model, initBase=True, initializeVelocity=True)
      #humanoid.resetPose()

      desiredPose = humanoid.computePose(humanoid._frameFraction)
      
      #desiredPose = desiredPose.GetPose()
      #curPose = HumanoidPoseInterpolator()
      #curPose.reset()
      
      s = humanoid.getState()
      #np.savetxt("pb_record_state_s.csv", s, delimiter=",")
      maxForces = [
          0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90, 90,
          90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100, 100,
          100, 60
      ]
      
      usePythonStablePD = True
      if usePythonStablePD:
        #taus = humanoid.computePDForces(desiredPose, desiredVelocities=None, maxForces=maxForces)
        taus = policy.run(s)
        #Print("taus=",taus)
        humanoid.convertActionToPose(taus)
      else:
        humanoid.computeAndApplyPDForces(desiredPose,maxForces=maxForces)

      pybullet_client.stepSimulation()
      t += 1. / 600.

  time.sleep(1. / 600.)
