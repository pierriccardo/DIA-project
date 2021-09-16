from numpy.lib.function_base import append
from pricing import PersonGenerator
import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm

from environment import PricingEnvironment, BiddingEnvironment, SpecificEnvironment
from context import ContextGenerator
from learners import *
from scipy.stats import norm, beta



class Experiment7_c():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates

        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        
        # bidding 
        self.bids = np.array(self.cm.bids)
        self.sigma = 10
        
        # self.means = self.cm.new_clicks(self.bids)
        
        
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.num_people = 10000*np.array(self.cm.class_distribution)

        self.T = 180 # number of days
        self.n_experiments = 10

        self.ret = self.cm.ret

        ## let's get pricing optima
        self.opt_pricing = np.zeros(3)
        self.opt = np.zeros(3)
        for macro in range(3):
            self.opt_pricing[macro] = np.max(self.cm.conv_rates[macro]*self.prices)
            means = self.cm.new_clicks_function_mean(self.bids, macro, self.num_people[macro])
            self.opt[macro] = np.max(means * ((self.ret[macro]+1)*self.opt_pricing[macro] - self.cm.mean_cc(self.bids)))
            #self.opt = np.max(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids)))




    def run(self):

        self.rewards_full = [[], [], []]
        pg = PersonGenerator()

        for e in tqdm(range(0, self.n_experiments)):

            Penv = MultiPricing(0)
            Benv = MultiBidding(0)

            for macro in range(3):
                Penv.add(PricingEnvironment(self.n_arms, self.cm.conv_rates[macro], self.prices))
                means = self.cm.new_clicks_function_mean(self.bids, macro, self.num_people[macro])
                sigmas = self.cm.new_clicks_function_sigma(self.bids, macro, self.num_people[macro])
                Benv.add(BiddingEnvironment(self.bids, means, sigmas))

            

            ts_learner = multi_TS_Learner(3, self.n_arms, self.prices)
            gpts_learner = multi_GPTS(3, self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = [[], [], []]

            return_times = [np.ones(1), np.ones(1), np.ones(1)]

            for t in range(0,self.T): # 1 round is one day

                
                pulled_prices, price_values = ts_learner.pull_arm()
                prices = self.prices[pulled_prices]
                
                for macro in range(3):
                    price_values[macro] *= np.mean(return_times[macro])
                
                pulled_bids = gpts_learner.pull_arm(price_values)

                news, cost = Benv.round(pulled_bids)

                buyers = Penv.round(pulled_prices, news)

                for macro in range(3):

                    new_returns = np.random.poisson(lam = 1+self.ret[macro], size = int(news[macro]))

                    reward = buyers[macro]*prices[macro]*np.mean(new_returns)-cost[macro]
                    not_buyer = news[macro]-buyers[macro]

                    ts_learner.update_more_single(macro, pulled_prices[macro], reward, buyers[macro], not_buyer)
                    gpts_learner.update_single(macro, pulled_bids[macro], news[macro])

                    rewards_this[macro].append(reward)

                    return_times[macro] = np.append(return_times[macro], new_returns)
            
            for macro in range(3):
                self.rewards_full[macro].append(rewards_this[macro])

    def plot(self):
        #sns.distplot(np.array(p_arms))
        
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        colors = ['r', 'g', 'b']
        labels = ['macro0', 'macro1', 'macro2']
        for macro in range(3):
            plt.plot(np.cumsum(np.mean(self.opt[macro] - self.rewards_full[macro], axis = 0)), colors[macro], label=labels[macro])
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_7_return_time.png")
        
        plt.show()

