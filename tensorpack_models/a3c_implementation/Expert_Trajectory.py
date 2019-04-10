class Experience(object):
    """
    Store values in the form (observation, action, reward) for each state in trajectory for n_trajectories
    """
    
    def __init__(self, trajectory_length = 10, n_trajectories = 5) -> None:
        self.trajectory_length = trajectory_length
        self.n_trajectories = n_trajectories
        self.observations = []
        self.actions = []
        self.rewards = []
        self.__curr_trajectory = -1
    
    def append(self, observation, action, reward) -> bool:
        import cv2, numpy as np
        
        observation = observation.transpose((3, 0, 1, 2))
        new_observation = []
        for i in range(len(observation)):
            new_observation.append(cv2.cvtColor(observation[i], cv2.COLOR_RGB2GRAY))
        new_observation = np.array(new_observation, dtype=np.uint8)
        observation = new_observation.transpose((2, 0, 1))
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

        if len(self.observations) == self.trajectory_length:
            return False
        
        return True

    def start_new_trajectory(self) -> bool:
        if self.__curr_trajectory == self.n_trajectories - 1:
            return False

        self.__curr_trajectory += 1
        self.observations = []
        self.actions = []
        self.rewards = []

        return True

    def save(self, prefix = '') -> None:
        import pickle, os

        assert len(self.observations) > 0, 'Please store expert trajectories before using this function'

        if not os.path.isdir('data'):
            os.mkdir('data')
        
        expert_data = {
            'paths': [{
                'observations': self.observations,
                'actions': self.actions,
                'rewards': self.rewards
            }]
        } 
        pickle.dump(expert_data, open('data/Assault/%s.pkl' % prefix, 'wb'))

    def fetch(self, prefix = '') -> dict:
        import pickle, os

        assert os.path.isdir('data'), 'Please save an expert trajectory before using this function'

        self.expert_data = pickle.load(open('data/%s.pkl' % prefix, 'rb'))
        return self.expert_data['paths']