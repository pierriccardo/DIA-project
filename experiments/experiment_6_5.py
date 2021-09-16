import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2, PricingEnvironment, BiddingEnvironment
from learners import *
from scipy.stats import norm, beta



class Experiment6new():
    
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

        # return prob #######################################################################################
        self.ret = self.cm.avg_ret
        
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

        self.T = 200 # number of days
        self.n_experiments = 10

    def run(self):

        self.rewards_full = []

        Benv = BidEnv2(self.bids, self.num_people)
        self.opt = Benv.compute_optimum(self.opt_pricing)[0]

        for e in tqdm(range(0, self.n_experiments)):
            
            Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)

            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            gpts_learner = GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = []

            past_costs = [np.array(0.44)]*self.n_arms

            return_times = np.ones(1)

            for t in range(0,self.T):
                 
                pulled_price, price_value = ts_learner.pull_arm()
                price = self.prices[pulled_price]

                mean_returns = np.mean(return_times)
                price_value *= mean_returns
                # il price value va moltiplicato per il numero di ritorni

                for bid in range(self.n_arms):  
                    gpts_learner.exp_cost[bid] = np.quantile(past_costs[bid], 0.8)
                
                pulled_bid = gpts_learner.pull_arm(price_value)

                news, costs = Benv.round(pulled_bid)
                news = int(news+0.5)
                buyer = Penv.round(pulled_price, news)

                new_returns = np.random.poisson(lam = self.ret, size = news)
                # simuliamo il numero di volte in cui ogni cliente ritorna

                pricing_rew = buyer*price-np.sum(costs)
                reward = news*self.opt_pricing-np.sum(costs)
                # il price è moltiplicato per il numero medio di volte in cui gli utenti sono tornati

                not_buyer = news-buyer

                past_costs[pulled_bid] = np.append(past_costs[pulled_bid], costs)
                
                ts_learner.update_more(pulled_price, pricing_rew, buyer, not_buyer)
                gpts_learner.update(pulled_bid, news)

                rewards_this.append(reward)

                return_times = np.append(return_times, new_returns)
            
            self.rewards_full.append(rewards_this)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g', label="GPTS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_6_new.png")

        #plt.show()