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
    
    def load_model(self):
        """ Load saved model into the IRL Tensorflow graph """ 
        tf.reset_default_graph()

        experts = load_latest_experts('data/'+self._env_name, n=3)
        irl_model = AIRLStateAction(env_spec = self._env.spec, expert_trajs = experts)
        policy = CategoricalConvPolicy(
            name='policy', 
            env_spec=self._env.spec,
            conv_filters=[32, 64, 64], 
            conv_filter_sizes=[3] * 3, 
            conv_strides=[2, 1, 2], 
            conv_pads=['SAME'] * 3
        )

        self._model = IRLTRPO(
            env = self._env,
            policy = policy,
            irl_model = irl_model,
            n_itr = N_ITERATIONS,
            batch_size = 200,
            max_path_length = 200,
            discount = 0.98,
            store_paths = True,
            discrim_train_itrs = 1,
            irl_model_wt = 1.0,
            entropy_weight = 0.1, # this should be 1.0 but 0.1 seems to work better
            zero_environment_reward = True,
            train_irl = False,
            baseline = ZeroBaseline(
                    env_spec = self._env.spec
                )
        )
        
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            saver = tf.train.Saver(model_vars)
            saver.restore(sess, '/home/tejas/Workspace/STRAP/models/irl/%s/model_%s' % (self._env_name, self._env_name))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.variables_initializer(model_vars))

        self.__reward_net = self._model.irl_model.discriminator
        print ('Model restored')
    
    def get_reward_network(self):
        """ Get the reward function to extract reward given Observation and Action values """
        return self.__reward_net

    def get_reward(self, observation, action):
        """ Pass Observation and Action, and get back reward from Reward Net """
        import numpy as np
        import tensorflow as tf
        
        observation = np.reshape(observation, (1, -1)).astype(np.float32)
        action_one_hot = np.zeros(self._env.action_space.n)
        action_one_hot[action-1] = 1
        action_one_hot = action_one_hot.reshape(1, -1)
        reward = None

        reward_op = self.__reward_net(self._env.spec, self._model.irl_model.obs_t, self._model.irl_model.act_t)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            reward = sess.run(reward_op, feed_dict={self._model.irl_model.obs_t: observation, self._model.irl_model.act_t: action_one_hot})

        return -reward

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------
Functions
--------------------------------------------------------------------------------------------------------------------------------------------------------
"""

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
    prediction_function = OfflinePredictor(PredictConfig(
        model=Model(env.action_space.n),
        session_init=get_model_loader("models/%s-v0.tfmodel" % env_name),
        input_names=['state'],
        output_names=['policy']
    ))
    play_n_episodes(env, prediction_function, DATA_COLLECT_EPISODES, render=True)
    


def train_AIRL(env_name):
    """ Train STRAP IRL Model """
    env = TfEnv(GymEnv(env_name+'-v0', record_video=False, record_log=False))
    
    experts = load_latest_experts('data/'+env_name, n=1)

    irl_model = AIRLStateAction(
        env_spec=env.spec, 
        expert_trajs=experts
    )
    policy = CategoricalConvPolicy(
        name='policy', 
        env_spec=env.spec,
        conv_filters=[32, 64, 64], 
        conv_filter_sizes=[3] * 3, 
        conv_strides=[2, 1, 2], 
        conv_pads=['SAME'] * 3
    )
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=N_ITERATIONS,
        batch_size=200,
        max_path_length=200,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=1,
        irl_model_wt=1.0,
        start_itr=START_ITR,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=ZeroBaseline(env_spec=env.spec)
    )

    losses = None
    saver = tf.train.Saver()
    with rllab_logdir(algo=algo, dirname='data/'+env_name+'_gcl'):
        with tf.Session() as sess:
            losses = algo.train()

            if not os.path.isdir('models/irl/%s' % env_name):
                os.mkdir('models/irl/%s' % env_name)
            saver.save(sess, '/home/tejas/Workspace/STRAP/models/irl/%s/model_%s' % (env_name, env_name))
    
    assert losses != None, "Please check implementation, loss not returned from training"
    with open('irl_loss_' + env_name + '.csv', 'a+') as loss_file:
        if START_ITR == 0:
            loss_file.write('Iteration,Loss\n')
        
        for itr, loss in enumerate(losses):
            loss_file.write('%d,%f\n' % (START_ITR + itr, loss))

    __plot_results(losses, "Change in loss over %d iterations" % len(losses))

def test_airl(env_name):
    """ Just a test function which passes initial environment observation and random action to STRAP IRL and gets some reward back """
    tfenv = TfEnv(GymEnv(env_name + '-v0', record_video=False, record_log=False))
    irl = IRL(tfenv, env_name)

    env = gym.make(env_name + '-v0')
    observation = irl.downsample_image(env.reset(), IMG_SIZE)
    print (observation.shape)
    observation = np.stack([observation] * STACK_SIZE, axis=2)
    action = np.asarray([env.action_space.sample()])
    reward = irl.get_reward(observation, action)
    print ('Reward = %d' % reward)