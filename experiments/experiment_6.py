import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm
from environment import PricingEnvironment, BiddingEnvironment
from learners import *
from scipy.stats import norm, beta



class Experiment6():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.num_people = 10000*np.array(self.cm.class_distribution)

        # self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        

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

        print(self.means)
        # print(self.means[indice])
        print(self.opt_pricing)
        # print(self.cm.mean_cc(self.bids)[indice])
        print(self.opt)
        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 2 # number of days
        self.n_experiments = 1

    def run(self):

        self.rewards_full = []

        for e in tqdm(range(0, self.n_experiments)):

            Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)

            Benv = BiddingEnvironment(self.bids, self.means, self.sigmas)

            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            gpts_learner = GPTS_learner_positive(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = []

            for t in range(0,self.T):
                 
                pulled_price, price_value = ts_learner.pull_arm()
                price = self.prices[pulled_price]

                pulled_bid = gpts_learner.pull_arm(price_value)

                # problema: il braccio tirato dal learner di bidding
                # deve dipendere da price
                
                news, cost = Benv.round(pulled_bid)
                buyer = Penv.round(pulled_price, news)

                reward = buyer*price-cost

                #print('empirical opt pricing: '+str(buyer*price/news)+' theoretical opt pricing: '+str(self.opt_pricing))
                
                not_buyer = news-buyer
                
                ts_learner.update_more(pulled_price, reward, buyer, not_buyer)
                gpts_learner.update(pulled_bid, news)

                rewards_this.append(reward)
            
            self.rewards_full.append(rewards_this)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g', label="GPTS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_6.png")

        #plt.show()

