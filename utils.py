"""
--------------------------------------------------------------------------------------------------------------------------------------------------------
Imports
--------------------------------------------------------------------------------------------------------------------------------------------------------
"""

''' Importing modules other folder '''
import sys
sys.path.append('inverse_rl/')
sys.path.append('rllab/')
sys.path.append('tensorpack_models/')

''' Tensorpack Data Collect '''
from tensorpack import OfflinePredictor, PredictConfig, get_model_loader
from tensorpack_models.a3c_implementation.expert import get_player, Model
from tensorpack_models.a3c_implementation.common import play_n_episodes

''' Sandbox imports '''
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy

''' RLLab imports '''
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

''' Inverse RL imports '''
from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.models.architectures import conv_net
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from inverse_rl.algos.irl_trpo import IRLTRPO

''' Other imports '''
import tensorflow as tf, gym, numpy as np, os, cv2
from scipy.misc import imresize
import vizdoomgym

''' Constants '''
from parameters import IMG_SIZE, STACK_SIZE, START_ITR, DATA_COLLECT_EPISODES, N_ITERATIONS

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------
Classes
--------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class InvalidArgumentError(Exception):
    """ Custom Exception defined for raising sys.argv arguments in main.py """
    pass

class IRL:
    """ STRAP Inverse Reinforcement Learning """
    def __init__(self, env, env_name):
        self._model = None
        self._env = env
        self._env_name = env_name
        self.load_model()

    def downsample_image(self, A, IM_SIZE, down_only=False, gray=True):
        """ Preprocess raw image """
        if down_only:
            obs = A
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            return imresize(obs,size=(IMG_SIZE, IMG_SIZE), interp= 'nearest')

        filt_size = 3
        blur_size = 5
        obs = A
        if gray:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs -= int(np.median(obs))
        _, obs = cv2.threshold(obs, 10, 255, cv2.THRESH_BINARY)

        obs = cv2.dilate(obs,np.ones((filt_size,filt_size), np.uint8), iterations=1)
        obs = cv2.GaussianBlur(obs,(blur_size, blur_size), 0)
        obs = cv2.GaussianBlur(obs,(blur_size, blur_size), 0)
        obs = imresize(obs,size=(IMG_SIZE, IMG_SIZE), interp= 'nearest')
        return obs
    
    def load_model(self, is_state_action = True):
        """ Load saved model into the IRL Tensorflow graph """ 

        """
        TODO: Check TF Graph
        """

        """ Load saved model into the IRL Tensorflow graph """ 
        tf.reset_default_graph()

        # Reconstruct entire AIRL graph
        self.reward_sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        if is_state_action:    
            saver = tf.train.import_meta_graph('../STRAP/models/irl/%s/model_%s.meta' % (self._env_name, self._env_name))
            saver.restore(self.reward_sess, tf.train.latest_checkpoint('../STRAP/models/irl/%s/' % (self._env_name)))

            # Get references to required tensors - Observation, action placeholders and energy operation
            operations = [t for op in tf.get_default_graph().get_operations() for t in op.values()]
            self.__observation = [t for t in operations if t.name == 'gcl/obs:0'][0]
            self.__action = [t for t in operations if t.name == 'gcl/act:0'][0]
            self.__reward_op = [t for t in operations if t.name == 'gcl/discrim/reward_fn:0'][0]
        else:
            saver = tf.train.import_meta_graph('../STRAP/models/irl_state/%s/model_%s.meta' % (self._env_name, self._env_name))
            saver.restore(self.reward_sess, tf.train.latest_checkpoint('../STRAP/models/irl_state/%s/' % (self._env_name)))

            # Get references to required operations - Observationplaceholders and reward operation
            operations = [t for op in tf.get_default_graph().get_operations() for t in op.values()]
            self.__observation = [t for t in operations if t.name == 'airl/obs:0'][0]
            self.__reward_op = [t for t in operations if t.name == 'airl/discrim/reward/fc_4/reward_fn/BiasAdd:0'][0]

        print ("------------- IRL Model restored -------------")
    
    def get_reward_network(self):
        """ Get the reward function to extract reward given Observation and Action values """
        return self.__reward_op

    def get_reward(self, observation, action):
        """ Pass Observation and Action, and get back reward from Reward Net """
        import numpy as np
        import tensorflow as tf
        
        observation = np.reshape(observation, (1, -1)).astype(np.float32)
        action_one_hot = np.zeros(self._env.action_space.n)
        action_one_hot[action-1] = 1
        action_one_hot = action_one_hot.reshape(1, -1)
        reward = None

        feed = {self.__observation: observation, self.__action: action_one_hot}
        reward = self.reward_sess.run(self.__reward_op, feed_dict=feed)
        return reward[0][0]

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------
Functions
--------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def get_device():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return '/gpu:0' if len([x.name for x in local_device_protos if x.device_type == 'GPU']) > 0 else '/cpu:0'

def __plot_results(losses, title=""):
    import matplotlib.pyplot as plt

    plt.plot([i + 1 for i in range(len(losses))], losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(title)
    plt.show()
    
def collect_data(env_name):
    """ Collect expert trajectories """
    env = get_player(env_name=env_name+'-v0', train=False)
    to_render = True

    try:
        env.render()
    except Exception:
        to_render = False

    prediction_function = OfflinePredictor(PredictConfig(
        model=Model(env.action_space.n),
        session_init=get_model_loader("models/%s-v0.tfmodel" % env_name),
        input_names=['state'],
        output_names=['policy']
    ))
    play_n_episodes(env, prediction_function, DATA_COLLECT_EPISODES, render=to_render)
    


def train_AIRL(env_name):
    """ Train STRAP IRL Model """
    env = TfEnv(GymEnv(env_name+'-v0', record_video=False, record_log=False))
    
    # AIRL Architecture
    experts = load_latest_experts('data/'+env_name, n=10)
    irl_model = AIRLStateAction(
        env_spec=env.spec, 
        expert_trajs=experts
    )
    # policy = CategoricalConvPolicy(
    #     name='policy', 
    #     env_spec=env.spec,
    #     conv_filters=[32, 64, 64], 
    #     conv_filter_sizes=[7, 5, 3], 
    #     conv_strides=[2, 1, 2], 
    #     conv_pads=['SAME'] * 3
    # )
    policy = CategoricalConvPolicy(
        name='policy', 
        env_spec=env.spec,
        conv_filters=[32, 64, 64], 
        conv_filter_sizes=[8, 4, 3], 
        conv_strides=[4, 2, 1], 
        conv_pads=['VALID'] * 3
    )
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=N_ITERATIONS,
        batch_size=50,
        max_path_length=500,
        discount=0.98,
        store_paths=True,
        discrim_train_itrs=1,
        irl_model_wt=1.0,
        start_itr=START_ITR,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=ZeroBaseline(env_spec=env.spec)
    )

    # Restoring previously saved model, if any, from latest checkpoint
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(model_vars)
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('models/irl'):
        os.mkdir('models/irl')
    if os.path.isdir('models/irl/%s' % env_name):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            saver.restore(sess, 'models/irl/%s/model_%s' % (env_name, env_name))
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            # sess.run(tf.variables_initializer(model_vars))
    else:
        os.mkdir('models/irl/%s' % env_name)
    print ("------------- Model restored -------------")

    # Training
    losses = None
    with rllab_logdir(algo=algo, dirname='data/'+env_name+'_gcl'):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            losses = algo.train()
            saver.save(sess, 'models/irl/%s/model_%s' % (env_name, env_name))
    
    # Writing losses to file
    assert losses != None, "BUG: Please check implementation, loss not returned from training"
    with open('results/irl_loss_' + env_name + '.csv', 'a+') as loss_file:
        if START_ITR == 0:
            loss_file.write('Iteration,Loss\n')
        
        for itr, loss in enumerate(losses):
            loss_file.write('%d,%f\n' % (START_ITR + itr, loss))

    __plot_results(losses, "Change in loss over %d iterations" % len(losses))

def test_airl(env_name):
    """ Just a test function which passes initial environment observation and random action to STRAP IRL and gets some reward back """
    import time

    tfenv = TfEnv(GymEnv(env_name + '-v0', record_video=False, record_log=False))
    irl = IRL(tfenv, env_name)

    env = gym.make(env_name + '-v0')
    start = time.time()
    observation = irl.downsample_image(env.reset(), IMG_SIZE, down_only=True)
    downsample_time = time.time() - start
    print ('Downsampling time: %.3f' % (time.time() - start))

    observation = np.stack([observation] * STACK_SIZE, axis=2)
    action = np.asarray([env.action_space.sample()])
    observation = observation.reshape((1, IMG_SIZE, IMG_SIZE, STACK_SIZE))
    start = time.time()
    reward = irl.get_reward(observation, action)
    print ('IRL time: %.3f' % (time.time() - start))

    print ('Reward = %d' % reward)
    irl.reward_sess.close()

def irl_stats(stat_to_show='IRLAverageEnergy'):
    import matplotlib.pyplot as plt
    import pandas as pd

    stats = pd.read_csv('results/progress.csv')
    vals = stats[stat_to_show]
    plt.plot(range(len(vals)), vals)
    plt.xlabel('Iteration')
    plt.ylabel(stat_to_show)
    plt.title('Analysis of IRL training')
    plt.show()