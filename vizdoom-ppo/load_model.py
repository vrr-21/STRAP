import gym
import vizdoomgym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('VizdoomTakeCover-v0')

env = DummyVecEnv([lambda: env])

model = PPO2.load("models/ppo_vizdoom_takecover")

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