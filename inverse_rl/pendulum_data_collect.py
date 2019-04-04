from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from inverse_rl.utils.log_utils import rllab_logdir
import vizdoomgym

def main():
    env = TfEnv((GymEnv('VizdoomTakeCover-v0', record_video=False, record_log=False)))
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=4,
        batch_size=100,
        max_path_length=20,
        discount=0.99,
        store_paths=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/vizdoom'):
        algo.train()

if __name__ == "__main__":
    main()
