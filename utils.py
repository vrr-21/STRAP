"""
--------------------------------------------------------------------------------------------------------------------------------------------------------
Imports
--------------------------------------------------------------------------------------------------------------------------------------------------------
"""

''' Importing modules from src/ folder '''
import sys
sys.path.append('inverse_rl/')

''' Sandbox imports '''
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy

''' RLLab imports '''
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

''' Inverse RL imports '''
from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.models.architectures import conv_net
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from inverse_rl.algos.irl_trpo import IRLTRPO

''' Other imports '''
from parameters import N_ITERATIONS
import tensorflow as tf, gym, numpy as np, os
from parameters import IMG_SIZE, STACK_SIZE, START_ITR

"""
--------------------------------------------------------------------------------------------------------------------------------------------------------
Classes
--------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class InvalidArgumentError(Exception):
    """ Custom Exception defined for raising sys.argv arguments in main.py """
    pass

class IRL:
    def __init__(self, env, env_name):
        self._model = None
        self._env = env
        self._env_name = env_name
        self.load_model()

    def __downsample_image(self, A, IM_SIZE):
        from scipy.misc import imresize

        B = A[31:195]
        B = B.mean(axis =2)
        B = B/255.0
        B = imresize(B, size= (IM_SIZE, IM_SIZE), interp= 'nearest')
        return B
    
    def load_model(self):        
        tf.reset_default_graph()

        experts = load_latest_experts('data/'+self._env_name, n=3)
        irl_model = AIRLStateAction(env_spec = self._env.spec, expert_trajs = experts)
        policy = CategoricalConvPolicy(
            name='policy', 
            env_spec=self._env.spec,
            conv_filters=[32,64,64], 
            conv_filter_sizes=[3]*3, 
            conv_strides=[2, 1, 2], 
            conv_pads=['SAME']*3
        )

        self._model = IRLTRPO(
            env = self._env,
            policy = policy,
            irl_model = irl_model,
            n_itr = N_ITERATIONS,
            batch_size = 100,
            max_path_length = 10,
            discount = 0.99,
            store_paths = True,
            discrim_train_itrs = 1,
            irl_model_wt = 1.0,
            entropy_weight = 0.1, # this should be 1.0 but 0.1 seems to work better
            zero_environment_reward = True,
            train_irl = False,
            baseline = LinearFeatureBaseline(
                    env_spec = self._env.spec
                )
        )
        
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            saver = tf.train.Saver(model_vars)
            saver.restore(sess, '/home/tejas/Workspace/STRAP/models/irl/model_'+self._env_name)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.variables_initializer(model_vars))

        self.__reward_net = self._model.irl_model.discriminator
        print ('Model restored')
    
    def get_reward_network(self):
        return self.__reward_net

    def get_reward(self, observation, action):
        import numpy as np
        import tensorflow as tf
        
        observation = np.reshape(self.__downsample_image(observation, IMG_SIZE), (1, -1)).astype(np.float32)
        action_one_hot = np.zeros(self._env.action_space.n)
        action_one_hot[action-1] = 1
        action_one_hot = action_one_hot.reshape(1, -1)
        reward = None

        reward_op = self.__reward_net(self._env.spec, observation, action_one_hot)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            reward = sess.run(reward_op, feed_dict={self._model.irl_model.obs_t: observation, self._model.irl_model.act_t: action_one_hot})

        return reward

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
    # tf.reset_default_graph()

    env = TfEnv((GymEnv(env_name+'-v0', record_video=False, record_log=False)))
    policy = CategoricalConvPolicy(
        name='policy', 
        env_spec=env.spec,
        conv_filters=[32,64,64], 
        conv_filter_sizes=[3]*3, 
        conv_strides=[2, 1, 2], 
        conv_pads=['SAME']*3
    )
    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=N_ITERATIONS,
        batch_size=20,
        max_path_length=20,
        discount=0.99,
        store_paths=True,
        start_itr=START_ITR,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )
    
    sess = tf.Session()
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.variables_initializer(model_vars))
    
    saver = tf.train.Saver(model_vars)
    if os.path.exists('models/expert'):
        saver.restore(sess, '/home/tejas/Workspace/STRAP/models/expert/model_'+env_name)
        print ('Loaded previously trained expert')
            
    losses = None
    with rllab_logdir(algo=algo, dirname='data/'+env_name):
        losses = algo.train()
        if not os.path.isdir('models/expert'):
            os.mkdir('models/expert')
        saver.save(sess, '/home/tejas/Workspace/STRAP/models/expert/model_'+env_name)

    assert losses != None, "Please check implementation, loss not returned from training"
    with open('data_collect_loss_' + env_name + '.csv', 'a+') as loss_file:
        if START_ITR == 0:
            loss_file.write('Iteration,Loss\n')

        for itr, loss in enumerate(losses):
            loss_file.write('%d,%f\n' % (itr, loss))
            
    print ('Losses saved')
    sess.close()

    # __plot_results(losses, "Change in loss over %d iterations" % len(losses))    


def train_AIRL(env_name):
    env = TfEnv(GymEnv(env_name+'-v0', record_video=False, record_log=False))
    
    experts = load_latest_experts('data/'+env_name, n=3)

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
        batch_size=100,
        max_path_length=20,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=1,
        irl_model_wt=1.0,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    losses = None
    saver = tf.train.Saver()
    with rllab_logdir(algo=algo, dirname='data/'+env_name+'_gcl'):
        with tf.Session() as sess:
            losses = algo.train()

            if not os.path.isdir('models/irl'):
                os.mkdir('models/irl')
            saver.save(sess, '/home/tejas/Workspace/STRAP/models/irl/model_'+env_name)
    
    assert losses != None, "Please check implementation, loss not returned from training"
    with open('irl_loss_' + env_name + '.csv', 'w+') as loss_file:
        loss_file.write('Iteration,Loss\n')
        for itr, loss in enumerate(losses):
            loss_file.write('%d,%f\n' % (itr, loss))

    # __plot_results(losses, "Change in loss over %d iterations" % len(losses))

def test_airl(env_name):
    tfenv = TfEnv(GymEnv(env_name + '-v0', record_video=False, record_log=False))
    irl = IRL(tfenv, env_name)

    env = gym.make(env_name + '-v0')
    observation = np.stack([env.reset()] * STACK_SIZE, axis=3)
    action = np.asarray([env.action_space.sample()])
    reward = irl.get_reward(observation, action)
    print ('Reward = %d' % reward)