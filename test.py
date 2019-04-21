lis = ['VizdoomBasic-v0',
'VizdoomCorridor-v0',
'VizdoomDefendCenter-v0',
'VizdoomDefendLine-v0',
'VizdoomHealthGathering-v0',
'VizdoomMyWayHome-v0',
'VizdoomPredictPosition-v0',
'VizdoomTakeCover-v0',
'VizdoomDeathmatch-v0',
'VizdoomHealthGatheringSupreme-v0']

import gym
import vizdoomgym

for item in lis:
    env = gym.make(item)
    env.reset()
    env.render()
    print ('Rending the env',item)
    k = 0
while (10000000):
    k+=1