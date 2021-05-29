import numpy as np
from learners import TS_Learner

class Context():
    """
    A context is defined by: 
    - an ID to identify it 
    - a learner algorithm
    - one or more subspaces of features 
    subspace is expressed as an array of tuple [('Y', 'I'),...]
    """

    def __init__(self, id, subspace, learner, obs=[], verbose=True):
        
        self.id = id
        self.subspace = subspace
        self.learner = learner
        self.obs = obs

    def update(self, features, pulled_arm, reward):
        # updates context learner
        self.learner.update(pulled_arm, reward)

        # update context observations
        self.obs.append([features, pulled_arm, reward])

        if self.verbose:
            print(f'[update | context id: {self.id} | new_obs-> features: {features}, pulled_arm: {pulled_arm}, reward: {reward}]')

    def train_sub_learner(self, obs, candidates):
        # train a new learner, using observations
        learner = TS_Learner(self.learner.n_arms, candidates)
        
        # train the learner
        for i, o in enumerate(obs):
            learner.update(o[i][1], o[i][2])

        return learner

        