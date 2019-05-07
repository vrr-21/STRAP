import gym
import vizdoomgym
import numpy as np
import sys

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from Expert_Trajectory import Experience

def update_state(state, obs):
   return np.append(state[:,:,1:], np.expand_dims(obs, 2), axis=2)

TRAJECTORY_LENGTH = 20000
N_TRAJECTORIES = 20

env_name = "VizdoomTakeCover-v0"
num_eps = 10
model_path = "models/ppo_VizdoomTakeCover-v0"
try:
    env_name = sys.argv[1]
    model_path = sys.argv[2]
    num_eps = int(sys.argv[3])
except:
    pass

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

model = PPO2.load(model_path)

experience = Experience(num_actions = env.action_space.n, trajectory_length= TRAJECTORY_LENGTH, n_trajectories= N_TRAJECTORIES)

obs = env.reset()
episode_rewards = [0.0]
current_ep = 1
env_name_stripped = env_name.split("-")[0]
state = np.stack([obs[0]]*4, axis=2)

while current_ep <= num_eps:
    action, _ = model.predict(obs)
    env.render()
    obs, rewards, dones, info = env.step(action)
    update_state(state, obs[0])
    episode_rewards[-1] += rewards[0]
    if not experience.append(state, action, rewards[0]):
        print("Episode {} reward: {}".format(current_ep, episode_rewards[-1]))
        experience.save(env_name = env_name_stripped, file_name='itr_'+str(current_ep - 1))
        episode_rewards.append(0.0)
        current_ep += 1
        obs = env.reset()
        state = np.stack([obs[0]]*4, axis=2)
        continue
    if dones[0]:
        print("Episode {} reward: {}".format(current_ep, episode_rewards[-1]))
        experience.save(env_name = env_name_stripped, file_name='itr_'+str(current_ep - 1))
        episode_rewards.append(0.0)
        current_ep += 1
        obs = env.reset()
        state = np.stack([obs[0]]*4, axis=2)
        continue