from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from inverse_rl.utils.log_utils import rllab_logdir
from parameters import N_ITERATIONS
import sys

def main():
    env_name = sys.argv[1]
    env = TfEnv((GymEnv(env_name+'-v0', record_video=False, record_log=False)))
    # policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    policy = CategoricalConvPolicy(
        name='policy', 
        env_spec=env.spec,
        conv_filters=[32,64,64], 
        conv_filter_sizes=[3]*3, 
        conv_strides=[1]*3, 
        conv_pads=['SAME']*3
    )

    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=N_ITERATIONS,
        batch_size=100,
        max_path_length=20,
        discount=0.99,
        store_paths=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/'+env_name):
        algo.train()

if __name__ == "__main__":
    main()
