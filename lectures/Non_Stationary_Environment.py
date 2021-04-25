from Environment import *

'''
A Non-Stationary Environment is an environment in 
which the arms reward functions are dependent from the 
current time 

we need to specify, for each arm, a reward function
which is dependent from the current time

we will define this class as an extension of the
environment class
'''

class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)

        # current round 
        self.t = 0

        self.horizon = horizon

    def round(self, pulled_arm):
        '''
        take as input the pulled arm and return the 
        reward, depending on the pulled arm and the 
        current pahse
        '''
        # num o phases is equal to the length of a 
        # row of the probability matrix
        n_phases = len(self.probabilities)
        phase_size = self.horizon / n_phases
        current_phase = int(self.t / phase_size)

        p = self.probabilities[current_phase][pulled_arm]
        self.t+=1
        return np.random.binomial(1, p)