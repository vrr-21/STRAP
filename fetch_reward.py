import tensorflow as tf
from utils import IRL, TfEnv, GymEnv
import gym
import vizdoomgym

class RewardCombine:
    def __init__(self, game_name):
        self.target_game_name = game_name
        self.target_actions = {
            0: "Turn left",
            1: "Turn right",
            2: "Move left",
            3: "Move right",
            4: "Shoot"
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
        action_name = self.target_actions[action]
        irl = None
        original_game_action = None

        if action_name in self.game_1_actions.keys():
            irl = self.irl_1
            original_game_action = self.game_1_actions[action_name]
            print("Action from %s" % (self.game_1))
        if action_name in self.game_2_actions.keys():
            irl = self.irl_2
            original_game_action = self.game_2_actions[action_name]
            print("Action from %s" % (self.game_2))
        
        reward = irl.get_reward(state, original_game_action)

        return reward