import sys
sys.path.append('inverse_rl/')
sys.path.append('rllab/')
sys.path.append('tensorpack_models/')
import tensorflow as tf
from utils import IRL, TfEnv, GymEnv
import gym
import vizdoomgym
import IPython
class RewardCombine:
    def __init__(self, game_name = "VizdoomCorridor-v0"):
        self.target_game_name = game_name
        self.target_actions = {
            0: "Move left",
            1: "Move right",
            2: "Shoot",
            3: "Move forward",
            4: "Move backward",
            5: "Turn left",
            6: "Turn right",
        }

        self.game_1 = "VizdoomTakeCover-v0"
        self.game_1_actions = {
            "Move left": 0,
            "Move right": 1
        }

        self.game_2 = "VizdoomDefendLine-v0"
        self.game_2_actions = {
            "Turn left": 0,
            "Turn right": 1,
            "Shoot": 2
        }
        
        self.irl_1 = IRL(TfEnv(GymEnv(self.game_1, record_video=False, record_log=False)), 'VizdoomTakeCover')

        self.irl_2 = IRL(TfEnv(GymEnv(self.game_2, record_video=False, record_log=False)), 'VizdoomDefendLine')

    def get_reward(self, state, action):
        # IPython.embed()
        action_name = self.target_actions[action]
        irl = None
        original_game_action = None
        game = -1
        if action_name in self.game_1_actions.keys():
            game = 1
        if action_name in self.game_2_actions.keys():
            game = 2
        if game > 0:
            if game == 1:
                irl = self.irl_1
                original_game_action = self.game_1_actions[action_name]
            else:
                irl = self.irl_2
                original_game_action = self.game_2_actions[action_name]
            reward = irl.get_reward(state, original_game_action)
            return reward, True
        else:
            reward = 0
            return reward, False