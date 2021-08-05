import numpy as np
import yaml
import random 
from configmanager import ConfigManager, new_clicks
'''
environment class is defined by:
- a number of arms
- a prob. distribution for each arm reward function


Environment interacts with the learner by returning 
a stochastic reward depending on the pulled arm
'''

class Environment():
    def __init__(self, n_arms, probabilities, candidates):

        self.candidates = candidates
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm): 

        reward = np.random.binomial(1, self.probabilities[pulled_arm]) * self.candidates[pulled_arm]
        return reward

class SpecificEnvironment():
    def __init__(self, n_arms, candidates):

        self.candidates = candidates
        self.n_arms = n_arms
        self.cm = ConfigManager()

    def round(self, pulled_arm, user_class):
        probabilities = self.cm.conv_rates[user_class]

        reward = np.random.binomial(1, probabilities[pulled_arm]) * self.candidates[pulled_arm]
        return reward

class BiddingEvironment():
  def __init__(self, bids, means, sigmas):
    self.bids = bids
    self.means = means
    self.sigmas = sigmas
    self.cm = ConfigManager()
  
  def cc(self, bid):
    return self.cm.cc(bid)         #(bid/(1+bid**0.5))

  def round(self, pulled_arm):    # pulled arm is the index of one of the bids
    news = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
    cost = news*self.cc(self.bids[pulled_arm])
    return news, cost
    # return news*(value - self.cc(self.bids[pulled_arm]))


class PricingEnvironment():
    def __init__(self, n_arms, probabilities, candidates):

        self.candidates = candidates
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm, num_clicks): # this time the number of people that click is determined by the bidding part,
        # thus, we cannot simply take one as before

        buyer = np.random.binomial(num_clicks, self.probabilities[pulled_arm])
        return buyer
   