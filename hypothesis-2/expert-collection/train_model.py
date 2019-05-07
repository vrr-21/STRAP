import gym
import vizdoomgym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

try:
    env = gym.make(sys.argv[1])
except:
    env = gym.make("VizdoomTakeCover-v0")

try:
    timesteps = int(sys.argv[1])
except:
    timesteps = 1000000

env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, ent_coef=0.5, verbose=1)

model.learn(total_timesteps=timesteps)

env_name = sys.argv[1]
# Save the agent
if not os.path.isdir('models/'):
    os.mkdir('models')
if not os.path.isdir('models/%s' % env_name):
    os.mkdir('models/%s' % env_name)

model.save('models/ppo_'+env_name)

print("Model saved.")