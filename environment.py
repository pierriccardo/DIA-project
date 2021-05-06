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

        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)

        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm): 
        
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
