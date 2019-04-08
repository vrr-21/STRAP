#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from builtins import range
import cv2


# In[2]:


import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# In[3]:


IM_SIZE = 80
IM_SIZE = 80


# In[4]:


CUDA_VISIBLE_DEVICES=0 #Using Nvidia GPU:0
TARGET_UPDATE_PERIOD = 100
MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50
K = 6 #action space


# In[5]:


def downsample_image(A,IMG_SIZE,down_only=False,gray=True):
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


# In[6]:

config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
tf.reset_default_graph()

sess_gan = tf.Session(config=config)
new_saver = tf.train.import_meta_graph('cyclegan-49.meta')
new_saver.restore(sess_gan, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
op_to_restore = graph.get_tensor_by_name("Model/g_A/t1:0")
ip1 = graph.get_tensor_by_name("input_A:0")
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()
# import IPython; IPython.embed()
# In[7]:


class DQN:
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope):

        self.K= K
        self.scope = scope

        with tf.variable_scope(scope):
            #considering input as 4 series of images
            self.X = tf.placeholder(tf.float32, shape = (None, 4, IM_SIZE, IM_SIZE), name = 'X')
            #order: (num_samples, height, width, "color")

            #RL variables
            self.G = tf.placeholder(tf.float32, shape = (None, ), name = 'G')
            self.actions = tf.placeholder(tf.int32, shape = (None, ), name = 'actions')

            #convolution
            Z =self.X/255.0
            Z= tf.transpose(Z, [0,2,3,1])

            for num_output_filters, filtersz, stridesz in conv_layer_sizes:
                #print("debugging: ")
                #print((num_output_filters, filtersz, stridesz))
                Z = tf.contrib.layers.conv2d(Z, num_output_filters, filtersz, stride = stridesz, activation_fn=tf.nn.relu)

            #fully connected layers
            Z = tf.contrib.layers.flatten(Z)
            for M in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z, M)

            #final layer
            self.predict_op = tf.contrib.layers.fully_connected(Z, K)

            #also one_hot_encode_all_predictions(actions)
            selected_action_values = tf.reduce_sum(self.predict_op* tf.one_hot(self.actions, self.K), reduction_indices=[1])

            cost = tf.reduce_sum(tf.square(self.G- selected_action_values))

            self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)

            self.cost = cost

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key= lambda x:x.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda x:x.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def set_session(self, session):
        self.session = session

    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states})

    def update(self, states, actions, targets):
        c,_ = self.session.run([self.cost, self.train_op], feed_dict = {self.X:states, self.G:targets, self.actions:actions})
        return c

    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])


# In[8]:


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
    #Sample experiences:
    samples = random.sample(experience_replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

    # Calculate the targets
    next_Qs= target_model.predict(next_states)

    #the reward with max reward:
    next_Q = np.max(next_Qs, axis = 1)

    #not including future state if the game is over
    targets = rewards + np.invert(dones).astype(np.float32)*gamma*(next_Q)

    #Update the model
    #Here current model is the learnig agent, target model is the temporarily stable one taking rest
    loss = model.update(states, actions, targets)
    return loss


# In[9]:


def play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_size, epsilon, epsilon_change, epsilon_min):
    t0 = datetime.now()

    #Reset the environment
    obs = env.reset()
    obs = downsample_image(obs,IMG_SIZE=256,gray=False)
    feed_dict = {ip1:np.expand_dims(obs,axis=0)}
    new_obs = sess_gan.run(op_to_restore,feed_dict)
    # import IPython; IPython.embed()
    new_obs = ((new_obs[0]+1)*127.5).astype(np.uint8)
    # new_obs = obs
    obs_small = downsample_image(new_obs,80,down_only=True)
    #always state is most recent 4 frames
    state = np.stack([obs_small]*4, axis =0)
    assert(state.shape == (4, 80, 80))
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
        obs_small = downsample_image(obs,80)

        next_state = np.append(state[1:],  np.expand_dims(obs_small, axis = 0), axis = 0)
        state  = next_state



        total_t += 1
        episode_reward += reward
        num_steps_in_episode += num_steps_in_episode


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


# In[10]:


def update_state(state, obs):
    obs_small = downsample_image(obs,80)
    return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)


# In[11]:


if __name__ == '__main__':
    #hyperparameters and initialization
    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    hidden_layer_sizes = [512]
    gamma = 0.99
    batch_size = 32
    num_episodes= 10
    total_t = 0
    experience_replay_buffer = []
    episode_rewards = np.zeros(num_episodes)


    #epsilon decays over time
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min)/ 500000

    #Make the environment
    env = gym.envs.make("DemonAttack-v0")
    #env = wrappers.Monitor(env, '../neural_reinforcement_agents')

    #Creat models
    model = DQN(K= K, conv_layer_sizes = conv_layer_sizes, hidden_layer_sizes = hidden_layer_sizes, gamma= gamma, scope = "model")

    target_model = DQN(K=K, conv_layer_sizes = conv_layer_sizes, hidden_layer_sizes = hidden_layer_sizes, gamma= gamma, scope = "target_model")


    # feed_dict ={ip1:np.expand_dims(img, axis=0)}


    # sess_reward = tf.InteractiveSession()

    with tf.Session(config=config) as sess:

        #Trying to fill in the experience buffer not learning anything!
        model.set_session(sess)
        target_model.set_session(sess)

        sess.run(tf.global_variables_initializer())

        obs = env.reset()
        obs_small = downsample_image(obs,80)

        state = np.stack([obs_small]*4, axis =0)

        for i in range(MIN_EXPERIENCES):
            action = np.random.choice(K)
            print('Experience: ', i)

            obs, reward, done,_ = env.step(action)
            feed_dict ={ip1:np.expand_dims(obs, axis=0)}
            # env.render()
            next_state = update_state(state, obs)

            experience_replay_buffer.append((state, action, reward, next_state, done))

            if done:
                obs = env.reset()
                obs_small = downsample_image(obs,80)
                state = np.stack([obs_small]*4, axis =0)

            else:
                state = next_state


        #Now play episodes and learning starts from here!!
        env = gym.envs.make("DemonAttack-v0")
        env = wrappers.Monitor(env, './', force= True)
        for i in range(num_episodes):

            total_t, episode_reward, duration, num_steps_in_episode, epsilon = play_one(env, total_t, experience_replay_buffer, model, target_model, gamma, batch_size , epsilon, epsilon_change, epsilon_min)
            episode_rewards[i] = episode_reward

            print("Episode:", i, " Duration:", duration, " Reward:", episode_reward)

            sys.stdout.flush()
    sess_gan.close()


# In[ ]:





# In[ ]:
