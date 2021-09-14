import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2, PriEnv
from learners import *
from scipy.stats import norm, beta



class Experiment7new():
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.num_people = 10000*np.array(self.cm.class_distribution)

        # self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        

        # bidding 
        self.bids = np.array(self.cm.bids)
        self.sigma = 10

        # return prob #######################################################################################
        self.ret = self.cm.avg_ret
        
        # self.means = self.cm.new_clicks(self.bids)
        self.means = self.cm.aggregated_new_clicks_function_mean(self.bids, self.num_people)
        self.sigmas = self.cm.aggregated_new_clicks_function_sigma(self.bids, self.num_people)

        print(self.means)
        # print(self.means[indice])
        print(self.opt_pricing)
        # print(self.cm.mean_cc(self.bids)[indice])
        print(self.opt)
        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 200 # number of days
        self.n_experiments = 10

    
    def run(self):

        self.rewards_full = []

        Benv_01 = BidEnv2(self.bids, self.num_people, classes=[0,1])
        Benv_23 = BidEnv2(self.bids, self.num_people, classes=[2,3])



        self.opt01 = Benv_01.compute_optimum(self.opt_pricing)[0]
        self.opt23 = Benv_23.compute_optimum(self.opt_pricing)[0]

        for e in tqdm(range(0, self.n_experiments)):
            
            Penv_01 = PriEnv(n_arms=self.n_arms, candidates=self.prices, classes = [0,1])            
            Penv_23 = PriEnv(n_arms=self.n_arms, candidates=self.prices, classes = [2,3])

            ts_learner01 = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            ts_learner23 = TS_Learner(n_arms=self.n_arms, candidates=self.prices)

            gpts_learner01 = GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2)
            gpts_learner23 = GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2)


