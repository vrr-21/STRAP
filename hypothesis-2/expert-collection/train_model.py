import gym
import vizdoomgym
import sys
import os
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env_name = "VizdoomTakeCover-v0"
timesteps = 1000000
try:
    env = gym.make(sys.argv[1])
    timesteps = int(sys.argv[2])
except:
    pass

env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=timesteps)

env_name = sys.argv[1]
# Save the agent
if not os.path.isdir('models/'):
    os.mkdir('models')

model.save('models/ppo_'+env_name)

print("Model saved.")