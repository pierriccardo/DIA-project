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


class BiddingEnvironment():
  def __init__(self, bids, means, sigmas, classe=0):
    self.bids = bids
    self.means = means
    self.sigmas = sigmas
    self.cm = ConfigManager()
    self.classe = classe
  

  def round(self, pulled_arm):    # pulled arm is the index of one of the bids
    news = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
    news = max(1.0,news)
    costs = self.cm.cost_per_click(self.bids[pulled_arm], self.classe, int(news+0.5))
    return news, costs
    

  def get_optimum(self, price_val, lam):
    best = -10000
    for pulled_arm in range(len(self.bids)):
      reward = self.means[pulled_arm]*(price_val*(lam+1)-self.cm.cost_per_click(self.bids[pulled_arm], self.classe, 0, mean = True))
      if (reward > best):
        best = reward
    return best


class PricingEnvironment():
    def __init__(self, n_arms, probabilities, candidates):

        self.candidates = candidates
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm, num_clicks): # this time the number of people that click is determined by the bidding part,
        # thus, we cannot simply take one as before

        buyer = np.random.binomial(num_clicks, self.probabilities[pulled_arm])
        return buyer

    def get_pricing_optimum(self):
      return max(self.probabilities*self.candidates)


##################################

########### materiale per Exp 7

##################################

class MultiBidding():
  def __init__(self, num, bids = 0, means = 0, sigmas = 0):

    self.num = num
    self.envs = []

    for i in range(num):
        self.envs.append(BiddingEnvironment(bids[i], means[i], sigmas[i]))
    
    self.cm = ConfigManager()

  def add(self, env):
      self.envs.append(env)
      self.num += 1
  
  def round(self, pulled_arm):    # pulled arm is the index of one of the bids
    news = []
    cost = []
    for i in range(self.num):
        ret = self.envs[i].round(pulled_arm[i])
        news.append(ret[0])
        cost.append(ret[1])
    return news, cost

class MultiPricing():
    def __init__(self, num, n_arms = 0, probabilities = 0, candidates = 0):

        self.num = num
        self.envs = []

        for i in range(num):
            self.envs.append(BiddingEnvironment(n_arms[i], probabilities[i], candidates[i]))

    def add(self, env):
       self.envs.append(env)
       self.num += 1

    def round(self, pulled_arm, num_clicks): # this time the number of people that click is determined by the bidding part,
        ret = []
        for i in range(self.num):
            ret.append(self.envs[i].round(pulled_arm[i], num_clicks[i]))
        return ret


###############################
# Nuovi environment con correzioni after gatti
###############################


class BidEnv2():
  def __init__(self, bids, num_people, classes = [0,1,2,3]):
    self.bids = bids
    self.cm = ConfigManager()
    self.classes = classes
    self.num_people = num_people

  def round(self, pulled_arm):
    total_news = 0
    costs = np.zeros(1)
    for c in self.classes:
      mean = self.cm.new_clicks_function_mean(self.bids[pulled_arm], c, self.num_people[c])
      std = self.cm.new_clicks_function_sigma(self.bids[pulled_arm], c, self.num_people[c])## per ulteriori test posso renderla 0
      news = np.random.normal(mean, std)
      #print(news)
      cost = self.cm.cost_per_click(self.bids[pulled_arm], c, size = int(news+0.5))
      total_news += news
      costs = np.append(costs, cost)
    return total_news, costs


  def compute_optimum(self, price_value, lam = 0):
    best = -10000
    best_arm = 0
    for pulled_arm in range(len(self.bids)):
      rew = 0
      for c in self.classes:

        mean = self.cm.new_clicks_function_mean(self.bids[pulled_arm], c, self.num_people[c])

        beta = np.sqrt(self.bids[pulled_arm])
        alpha = self.cm.costo[c]
        cost = self.bids[pulled_arm] * alpha /( beta + alpha )

        rew += mean*((lam+1)*price_value-cost)
      
      

      if(rew > best):
        best = rew
        best_arm = pulled_arm
    
    return best, best_arm



class PriEnv():
    def __init__(self, n_arms, candidates, classes = [0,1,2,3]):

        self.candidates = candidates
        self.n_arms = n_arms
        self.classes = classes
        self.cm = ConfigManager()
        self.total_recall = 0
        for c in classes:
          self.total_recall += self.cm.class_distribution[c]

        # self.probabilities = self.cm.conv_rates
    
    def round(self, pulled_arm, num_clicks):
      buyer = 0
      for c in self.classes:
        clicks = num_clicks*self.cm.class_distribution[c]/self.total_recall
        buyer += np.random.binomial(clicks, self.cm.conv_rates[c][pulled_arm])
      
      return buyer

    def compute_optimum(self):
      best = -10000
      idx = 0
      for pulled_arm in range(self.n_arms):
        reward = 0
        for c in self.classes:
          reward += self.cm.conv_rates[c][pulled_arm]*self.cm.class_distribution[c]*self.candidates[pulled_arm]
        if(reward>best):
          best = reward
          idx = pulled_arm
      return best, idx

