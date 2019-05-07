import gym
import vizdoomgym
import numpy as np
import sys

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env_name = 'VizdoomTakeCover-v0'
model_path = "models/ppo_VizdoomTakeCover-v0.pkl"
try:
    env_name = sys.argv[1]
    model_path = sys.argv[2]
except:
    pass

env = gym.make(env_name)

env = DummyVecEnv([lambda: env])

model = PPO2.load(model_path)

episode_rewards = [0.0]
obs = env.reset()
num_eps = 3
current_ep = 1
while current_ep <= num_eps:
    action, _states = model.predict(obs)
    env.render()
    obs, rewards, dones, info = env.step(action)
    episode_rewards[-1] += rewards[0]
    if dones[0]:
        obs = env.reset()
        print("Episode {} reward: {}".format(current_ep, episode_rewards[-1]))
        episode_rewards.append(0.0)
        current_ep += 1