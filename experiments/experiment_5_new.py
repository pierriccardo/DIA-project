import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape, size

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2, BiddingEnvironment, PricingEnvironment
from learners import *
from scipy.stats import norm, beta


class Experiment5new():
    
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
        self.sigma = 2
        # self.p = np.array([.1, .03, .34, .28, .12, .19, .05, .56, .26, .35]) # cm.aggr_conv_rates()
        # perchè ridefiniamo questa???
        
        # self.means = self.cm.new_clicks(self.bids)
        self.means = self.cm.aggregated_new_clicks_function_mean(self.bids, self.num_people)
        self.sigmas = self.cm.aggregated_new_clicks_function_sigma(self.bids, self.num_people)

        self.opt = np.max(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids, classes= [0,1,2,3])))
        indice = np.argmax(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids, classes= [0,1,2,3])))

        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 365 # number of days
        self.n_experiments = 10

    def run(self):
        print(self.cm.new_clicks)
        
        print(self.cm.cost_per_click)
        Benv = BidEnv2(self.bids, self.num_people)
        self.opt = Benv.compute_optimum(self.opt_pricing)[0]
        print(self.opt)

        self.rewards_full = []

        self.com_reg = []


        for e in tqdm(range(0, self.n_experiments)):

            gpts_learner = GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = []

            past_costs = [np.array(0.44)]*self.n_arms
            

            for t in range(0,self.T):
                 
                pulled_price = self.opt_arm_pricing # lo cambio come voglio
                price_value = self.opt_pricing
                price = self.prices[pulled_price]

                for bid in range(self.n_arms):  ## update quantiles of expected costs
                    gpts_learner.exp_cost[bid] = np.quantile(past_costs[bid], 0.8)# np.mean(past_costs[bid])

                pulled_bid = gpts_learner.pull_arm(price_value)

                # problema: il braccio tirato dal learner di bidding
                # deve dipendere da price
                
                news, costs = Benv.round(pulled_bid)
                
                reward = news*self.opt_pricing-np.sum(costs)

                print('reward: '+str(reward)+' ottimo teorico: '+str(self.opt))
                past_costs[pulled_bid] = np.append(past_costs[pulled_bid], costs)
                
                gpts_learner.update(pulled_bid, news)

                rewards_this.append(reward)
            
            self.rewards_full.append(rewards_this)
            self.com_reg.append(np.cumsum((self.opt - self.rewards_full)))

        


    def plot(self):
        #sns.distplot(np.array(p_arms))
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g', label="GPTS")
        plt.plot(np.quantile(np.cumsum(self.opt - self.rewards_full, axis=1), q=0.025, axis = 0),'g',linestyle='dashed', label="GPTS Confidence Interval 95%")
        plt.plot(np.quantile(np.cumsum(self.opt - self.rewards_full, axis=1), q=0.975,  axis = 0),'g',linestyle='dashed')
        #plt.plot(1.96*np.cumsum(np.std(self.opt - self.rewards_full, axis=0))/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g',linestyle='dashed', label="GPTS Confidence Interval 95%")
        #plt.plot(-1.96*np.cumsum(np.std(self.opt - self.rewards_full, axis=0))/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g',linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_5_new.png")
        #plt.show()