import numpy as np
import yaml
import random 
'''
environment class is defined by:
- a number of arms
- a prob. distribution for each arm reward function


Environment interacts with the learner by returning 
a stochastic reward depending on the pulled arm
'''

class Environment():
    def __init__(self, n_arms, probabilities):

        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm): 
        
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

class SpecificEnvironment():
    def __init__(self, n_arms, probabilities, candidates):

        self.candidates = candidates
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm): 

        reward = np.random.binomial(1, self.probabilities[pulled_arm]) * self.candidates[pulled_arm]
        return reward
    