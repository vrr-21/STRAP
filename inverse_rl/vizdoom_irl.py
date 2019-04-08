import tensorflow as tf

from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from inverse_rl.models.architectures import conv_net


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts

from parameters import N_ITERATIONS
import sys

def main():
    env_name = sys.argv[1]
    env = TfEnv(GymEnv(env_name+'-v0', record_video=False, record_log=False))

    experts = load_latest_experts('data/'+env_name, n=5)

    irl_model = AIRLStateAction(
        env_spec=env.spec,
        expert_trajs=experts
    )

    policy = CategoricalConvPolicy(
        name='policy',
        env_spec=env.spec,
        conv_filters=[32, 64, 64],
        conv_filter_sizes=[3] * 3,
        conv_strides=[1] * 3,
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
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=ZeroBaseline(env_spec=env.spec)
    )

    saver = tf.train.Saver()
    with rllab_logdir(algo=algo, dirname='data/'+env_name+'_gcl'):
        with tf.Session() as sess:
            algo.train()
            saver.save(sess, './model_'+env_name)

if __name__ == "__main__":
    main()
