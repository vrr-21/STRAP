import gym
import vizdoomgym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from Expert_Trajectory import Experience

TRAJECTORY_LENGTH = 20000
N_TRAJECTORIES = 20

env = gym.make('VizdoomTakeCover-v0')

env = DummyVecEnv([lambda: env])

model = PPO2.load("models/ppo_vizdoom_takecover-colab")

experience = Experience(num_actions = env.action_space.n, trajectory_length= TRAJECTORY_LENGTH, n_trajectories= N_TRAJECTORIES)

obs = env.reset()
episode_rewards = [0.0]
current_ep = 1
num_eps = 10
while current_ep <= num_eps:
    action, _ = model.predict(obs)
    env.render()
    obs_cache = obs
    obs, rewards, dones, info = env.step(action)
    episode_rewards[-1] += rewards[0]
    if not experience.append(obs_cache, action, rewards[0]):
        print("Episode {} reward: {}".format(current_ep, episode_rewards[-1]))
        experience.save(env_name = 'VizDoomTakeCoverColab', file_name='itr'+str(current_ep - 1))
        episode_rewards.append(0.0)
        current_ep += 1
        obs = env.reset()
        continue
    if dones[0]:
        print("Episode {} reward: {}".format(current_ep, episode_rewards[-1]))
        experience.save(env_name = 'VizDoomTakeCoverColab', file_name='itr'+str(current_ep - 1))
        episode_rewards.append(0.0)
        current_ep += 1
        obs = env.reset()
        continue