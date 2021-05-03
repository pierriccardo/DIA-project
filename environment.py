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

    def round(self, pulled_arm): # our pulled arm is the price
        bid = 0.42

        alpha = self.config["aggregated"]["cost_per_click"]
        reward = self.probabilities[pulled_arm] - self.cost_per_click(bid, alpha)
        
        #reward = np.random.binomial(1, self.probabilities[pulled_arm])
        #print(reward)
        return reward / self.probabilities[pulled_arm]

    def conv_rate(self, x, a=1, b=1, c=1):
        return ((c*x) ** a) * np.exp(-b * c * x)

    def cost_per_click(self, bid, alpha):
        beta = np.sqrt(bid)
        return bid * np.random.beta(alpha, beta, 1)

    def return_probability(self, _lambda, size=100000):
        return np.random.poisson(_lambda, size=size)

    def new_clicks(self, bid, Na=10000, p0=0.01, cc=0.44):
        p = 1-(cc/(2*bid))
        mean = Na*p*p0
        sd = (Na*p*p0*(1-p0))**0.5
        return np.random.normal(mean,sd)