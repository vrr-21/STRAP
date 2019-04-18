import sys
sys.path.append('inverse_rl/')
sys.path.append('rllab/')
sys.path.append('inverse_rl/vizdoomgym')

from utils import collect_data, train_AIRL, test_airl, test_dqn, InvalidArgumentError

if __name__ == "__main__":

    n_arguments = len(sys.argv)
    if n_arguments < 3:
        if n_arguments < 2:
            raise InvalidArgumentError("No environment specified")
        raise InvalidArgumentError("Please specify mode of operation")

    env = sys.argv[1]
    mode = sys.argv[2].lower()
    # env = "Assault"
    # mode = "irl"

    if mode == 'collect':
        collect_data(env)
    elif mode == 'irl':
        train_AIRL(env)
    elif mode == 'test':
        test_airl(env)
    elif mode == 'dqn':
        test_dqn()
    else:
        raise InvalidArgumentError("Mode '" + mode + "' not supported.")