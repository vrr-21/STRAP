class Experience(object):
    """
    Store values in the form (observation, action, reward) for each state in trajectory for n_trajectories
    """
    
    def __init__(self, num_actions, trajectory_length = 10, n_trajectories = 5):
        self.trajectory_length = trajectory_length
        self.n_trajectories = n_trajectories
        self.observations = []
        self.actions = []
        self.rewards = []
        self.__curr_trajectory = -1
        self.num_actions = num_actions
    
    def downsample_image(self, image):
        import cv2, numpy as np
        # from parameters import IMG_SIZE
        IMG_SIZE = 84

        new_image = []
        for i in range(len(image)):
            new_image.append(cv2.resize(cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY), (IMG_SIZE, IMG_SIZE)))

        new_image = np.array(new_image, dtype=np.uint8)
        return new_image

    def append(self, observation, action, reward):
        import cv2, numpy as np
        
        observation = self.downsample_image(observation)
        self.observations.append(observation.reshape(-1))

        action_one_hot = np.zeros(self.num_actions)
        action_one_hot[action - 1] = 1
        self.actions.append(action_one_hot)
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

    def save(self, env_name = 'Assault', file_name = 'itr'):
        import pickle, os

        assert len(self.observations) > 0, 'Please store expert trajectories before using this function'

        if not os.path.isdir('data'):
            os.mkdir('data')
        if not os.path.isdir('data/%s' % env_name):
            os.mkdir('data/%s' % env_name)
        
        expert_data = {
            'paths': [{
                'observations': self.observations,
                'actions': self.actions,
                'rewards': self.rewards
            }]
        } 

        pickle.dump(expert_data, open('data/%s/%s.pkl' % (env_name, file_name), 'wb+'))

    def fetch(self, prefix = ''):
        import pickle, os

        assert os.path.isdir('data'), 'Please save an expert trajectory before using this function'

        self.expert_data = pickle.load(open('data/%s.pkl' % prefix, 'rb'))
        return self.expert_data['paths']
