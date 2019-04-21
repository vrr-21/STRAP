import tensorflow as tf
import gym
import vizdoomgym
import numpy as np
import cv2
import random
from parameters import IMG_SIZE
from fetch_reward import RewardCombine
from dqn import DQN
from scipy.misc import imresize
from datetime import datetime
import sys

# Constants
CUDA_VISIBLE_DEVICES=0 #Using Nvidia GPU:0
TARGET_UPDATE_PERIOD = 100
MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 32
K = 5 #action space

def downsample_image(A, IMG_SIZE = 84, down_only=False, gray=True):
    if down_only:
        obs = A
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        return imresize(obs,size=(IMG_SIZE,IMG_SIZE),interp= 'nearest')
    filt_size = 3
    blur_size = 5
    # cv2_imshow(obs)
    obs = A
    if gray:
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # subtarcting median pizel
    obs -= int(np.median(obs))
    # obs.convertTo(obs,CV_8UC1)
    # binarizing
    _,obs = cv2.threshold(obs,10,255,cv2.THRESH_BINARY)

    obs = cv2.dilate(obs,np.ones((filt_size,filt_size), np.uint8),iterations=1)
    obs = cv2.GaussianBlur(obs,(blur_size,blur_size),0)
    obs = cv2.GaussianBlur(obs,(blur_size,blur_size),0)
    obs = imresize(obs,size=(IMG_SIZE,IMG_SIZE),interp= 'nearest')
    # print (obs.shape)
    # cv2_imshow(obs)
    # cv2.waitKey() #image will not show until this is called
    # cv2.destroyWindow('HelloWorld') #make sure window closes cleanly
    return obs

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    #Sample experiences:
    samples = random.sample(experience_replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

    # rewards = rewards.transpose([1,0])

    # Calculate the targets
    next_Qs= target_model.predict(next_states)
    # import IPython; IPython.embed();

    #the reward with max reward:
    next_Q = np.max(next_Qs, axis = 1)

    #not including future state if the game is over
    targets = rewards + np.invert(dones).astype(np.float32)*gamma*(next_Q)

    #Update the model
    #Here current model is the learnig agent, target model is the temporarily stable one taking rest
    loss = model.update(states, actions, targets)
    return loss

def play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_size, epsilon, epsilon_change, epsilon_min, episode_num=0):
    import time
    t0 = datetime.now()
    r = RewardCombine("VizdoomCorridor-v0")
    #Reset the environment
    obs = env.reset()
    obs_small = downsample_image(obs, IMG_SIZE, down_only=True)
    #always state is most recent 4 frames
    state = np.stack([obs_small]*4, axis=2)
    assert(state.shape == (IMG_SIZE, IMG_SIZE, 4))
    loss = None
        
    total_time_training = 0
    num_steps_in_episode = 1
    episode_reward = 0
    #env = env.monitor.start('../neural_reinforcement_agents')
    done = False
    while not done:
        #Update the current training network

        if total_t % TARGET_UPDATE_PERIOD == 0:
            #periodically save the current learnings into a temp copy to bring extra stability
            target_model.copy_from(model)

        #also take actions to learn the game
        action = model.sample_action(state, epsilon)

        #find the reward
        obs, reward, done,_ = env.step(action)

        obs_small = downsample_image(obs, IMG_SIZE, down_only=True)
        reward_irl = r.get_reward(state, action)
        

        if to_render:
            env.render()

        next_state = np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis=2)
        state = next_state

        total_t += 1
        print ('Episode: %d, Iteration: %d, IRL Reward: %.3f, Env reward: %d' % (episode_num, total_t, reward_irl, reward_env))
        episode_reward += reward_irl
        num_steps_in_episode += num_steps_in_episode

        reward = reward_irl

        #updating the experience replay buffer
        if len(experience_replay_buffer)> MAX_EXPERIENCES:
            experience_replay_buffer.pop(0)

        experience_replay_buffer.append((state, action, reward, next_state, done))

        #train the model
        t0_2 = datetime.now()
        
        loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        
        dt = datetime.now() - t0_2

        total_time_training += dt.total_seconds()

        #updating the epsilon value
        epsilon = max(epsilon- epsilon_change, epsilon_min)

    return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, epsilon

def update_state(state, obs):
    return np.append(state[:,:,1:], np.expand_dims(obs, 2), axis=2)

if __name__ == "__main__":
    game_name = "VizdoomCorridor-v0"
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1), (32, 3, 1)]
    hidden_layer_sizes = [512, 256]
    gamma = 0.99
    batch_size = 32
    num_episodes= int(sys.argv[1])
    total_t = 0
    experience_replay_buffer = []
    episode_rewards = np.zeros(num_episodes)
    reward_fetcher = RewardCombine(game_name)

    #epsilon decays over time
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min)/ 500000

    env = gym.make(game_name)
    to_render = False

    try:
        if to_render:
            env.render()
    except:
        to_render = False

    model = DQN(K= K, conv_layer_sizes = conv_layer_sizes, hidden_layer_sizes = hidden_layer_sizes, gamma= gamma, scope = "model")
    target_model = DQN(K=K, conv_layer_sizes = conv_layer_sizes, hidden_layer_sizes = hidden_layer_sizes, gamma= gamma, scope = "target_model")    

    # with tf.Session() as sess_dqn:
    sess_dqn = tf.Session()

    #Trying to fill in the experience buffer not learning anything!
    model.set_session(sess_dqn)
    target_model.set_session(sess_dqn)

    sess_dqn.run(tf.global_variables_initializer())

    obs = env.reset()
    obs_small = downsample_image(obs, IMG_SIZE, down_only= True)

    state = np.stack([obs_small]*4, axis=2)

    for i in range(MIN_EXPERIENCES):
        print("Starting experience gathering:")
        action = np.random.choice(K)
        print('Experience: ', i)
        
        irl_reward = reward_fetcher.get_reward(state, action)

        obs, reward_env, done,_ = env.step(action)
        obs_small = downsample_image(obs, IMG_SIZE, down_only= True)

        print("\t--- IRL Reward: {} ----- Env Reward: {}".format(irl_reward, reward_env))

        if to_render:
            env.render()
        print ('Experience collected: %3d/%3d' % (i, MIN_EXPERIENCES), end='\r')
        next_state = update_state(state, obs_small)

        # import IPython; IPython.embed();
        experience_replay_buffer.append((state, action, irl_reward, next_state, done))

        if done:
            obs = env.reset()
            obs_small = irl.downsample_image(obs, IMG_SIZE)
            state = np.stack([obs]*4, axis=2)

        else:
            state = next_state
    
    saver = tf.train.Saver()

    for i in range(num_episodes):
        total_t, episode_reward, duration, num_steps_in_episode, epsilon = play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_size , epsilon, epsilon_change, epsilon_min, i)
        episode_rewards[i] = episode_reward

        print("Episode:", i, " Duration:", duration, " Reward:", episode_reward)
        sys.stdout.flush()
    saver.save(sess_dqn, 'models/dqn/VizdoomCombined/VizdoomCombined')
    print ('--------------- Model saved afer %d episodes ---------------' % num_episodes)

    sess_dqn.close()
