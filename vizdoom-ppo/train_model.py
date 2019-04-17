import gym
import vizdoomgym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('VizdoomTakeCover-v0')

env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, ent_coef=0.5, verbose=1)

model.learn(total_timesteps=100000)
# Save the agent
model.save("models/ppo_vizdoom_takecover1e6")

print("Model saved.")