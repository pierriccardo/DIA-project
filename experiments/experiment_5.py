import numpy as np
from configmanager import ConfigManager
from tqdm import tqdm
from environment import PricingEnvironment
from environment import BiddingEvironment
from learners import *
from scipy.stats import norm, beta
import matplotlib.pyplot as plt


class Experiment5():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.num_people = 10000*np.array(self.cm.class_distribution)

        # self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        self.opt_arm_pricing = np.argmax(np.multiply(self.p, self.prices))
        

        # bidding 
        self.bids = np.array(self.cm.bids)
        self.sigma = 10
        # self.p = np.array([.1, .03, .34, .28, .12, .19, .05, .56, .26, .35]) # cm.aggr_conv_rates()
        # perchè ridefiniamo questa???
        
        # self.means = self.cm.new_clicks(self.bids)
        self.means = self.cm.aggregated_new_clicks_function_mean(self.bids, self.num_people)
        self.sigmas = self.cm.aggregated_new_clicks_function_sigma(self.bids, self.num_people)


        

        self.opt = np.max(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids)))
        indice = np.argmax(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids)))

        print(self.means[indice])
        print(self.opt_pricing)
        print(self.cm.mean_cc(self.bids)[indice])
        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 200 # number of days
        self.n_experiments = 8

    def run(self):

        self.rewards_full = []

        for e in tqdm(range(0, self.n_experiments)):

            Benv = BiddingEvironment(self.bids, self.means, self.sigmas)
            

            gpts_learner = GPTS_learner_positive(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = []

            for t in range(0,self.T):
                 
                pulled_price = self.opt_arm_pricing # lo cambio come voglio
                price_value = self.opt_pricing
                price = self.prices[pulled_price]

                pulled_bid = gpts_learner.pull_arm(price_value)

                # problema: il braccio tirato dal learner di bidding
                # deve dipendere da price
                
                news, cost = Benv.round(pulled_bid)
                
                reward = news*self.opt_pricing-cost

                #print('empirical opt pricing: '+str(buyer*price/news)+' theoretical opt pricing: '+str(self.opt_pricing))
                
                
                gpts_learner.update(pulled_bid, news)

                rewards_this.append(reward)
            
            self.rewards_full.append(rewards_this)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        
        
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g')
        plt.legend(["GPTS"])
        plt.savefig("img/experiments/experiment_5.png")
        

        #plt.show()