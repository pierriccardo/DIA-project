from learners.TS_Learner import *


'''
'''

class SWTS_Learner(TS_Learner):

    # window_size will be used to consider
    # recent bservations on the beta parameters
    # extimation
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size

    def update(self, pulled_arm, reward):
        self.t+=1
        self.update_observations(pulled_arm, reward)
        
        # beta params are updated by computeing 2 values:

        # 1: cumulative reward obtained by the pulled arm in the last rounds
        # were the num of obs used is the sliding window size
        cum_rew = np.sum(self.rewards_per_arm[pulled_arm][-self.window_size:])

        # 2: number of times that we pulled the arm
        n_rounds_arm = len(self.rewards_per_arm[pulled_arm][-self.window_size:])

        self.beta_parameters[pulled_arm, 0] = cum_rew + 1.0
        self.beta_parameters[pulled_arm, 1] = n_rounds_arm - cum_rew + 1.0