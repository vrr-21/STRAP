#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu

import cv2
import gym
import multiprocessing as mp
import numpy as np
import os
import six
import sys
import uuid
import tensorflow as tf

from tensorpack import *

from .atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState
from .common import Evaluator, eval_model_multithread, play_n_episodes
from .simulator import SimulatorMaster, SimulatorProcess, TransitionExperience
from parameters import STACK_SIZE, IMG_SIZE

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)
GAMMA = 0.99
STATE_SHAPE = IMAGE_SIZE + (3, )

# NUM_ACTIONS = None
ENV_NAME = None


def get_player(train=False, dumpdir=None, env_name='Assault-v0'):
    env = gym.make(env_name)
    if dumpdir:
        env = gym.wrappers.Monitor(env, dumpdir, video_callable=lambda _: True)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    env = FrameStack(env, 4)
    if train:
        env = LimitLength(env, 60000)
    return env


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def inputs(self):
        assert self.num_actions is not None
        return [tf.TensorSpec((None,) + STATE_SHAPE + (STACK_SIZE, ), tf.uint8, 'state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'futurereward'),
                tf.TensorSpec((None,), tf.float32, 'action_prob'),
                ]

    def _get_NN_prediction(self, state):
        assert state.shape.rank == 5  # Batch, H, W, Channel, History
        state = tf.transpose(state, [0, 1, 2, 4, 3])  # swap channel & history, to be compatible with old models
        image = tf.reshape(state, [-1] + list(STATE_SHAPE[:2]) + [STATE_SHAPE[2] * STACK_SIZE])

        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('conv0', image, 32, 5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, 32, 5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, 64, 4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, 64, 3)

        l = FullyConnected('fc0', l, 512)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, self.num_actions)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1)
        return logits, value

    def build_graph(self, state, action, futurereward, action_prob):
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, self.num_actions), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, self.num_actions), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   cost, tf.reduce_mean(importance, name='importance'))
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt

class A3CExpert:
    def __init__(self, env_name):
        self.env_name = env_name
        self.num_actions = get_player(env_name=self.env_name).action_space.n
        self.prediction_function = OfflinePredictor(PredictConfig(
            model=Model(self.num_actions),
            session_init=get_model_loader("Assault-v0.tfmodel"),
            input_names=['state'],
            output_names=['policy']))
        self.env = get_player(train=False, env_name=self.env_name)
    
    def get_optimal_action(self, observation):
        observation = np.expand_dims(observation, 0)
        return self.prediction_function(observation)[0][0].argmax()

    def get_action_probs(self, observation):
        observation = np.expand_dims(observation, 0)
        return self.prediction_function(observation)[0][0]

# def collect():
#     model = A3CExpert('Assault-v0')
#     play_n_episodes(get_player(train=False), model.prediction_function, 1, render=True)