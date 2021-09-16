import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2, PricingEnvironment, BiddingEnvironment
from learners import *
from scipy.stats import norm, beta



class Experiment6():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.num_people = self.cm.num_people*np.array(self.cm.class_distribution)

        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices)
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        

        # bidding 
        self.bids = np.array(self.cm.bids)

        # calcola i ritorni medi delle varie classi
        self.ret = self.cm.aggr_return_probability([0,1,2,3])
        self.ret = 1 #self.cm.aggr_return_probability([0,1,2,3])
        
    
        #self.gpts_reward_per_experiment = []
        #self.ts_reward_per_experiments = []
        #self.p_arms = []

        self.T = 200 # number of days
        self.n_experiments = 3

    def run(self):

        self.rewards_full = [] #lista delle reward dei vari esperimenti 

        Benv = BidEnv2(self.bids, self.num_people)
        Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
        self.opt = Benv.compute_optimum(self.opt_pricing, self.ret)[0]

        for e in tqdm(range(0, self.n_experiments)):
            
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            gpts_learner = GPTS(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = [] # reward dell'esperimento 

            past_costs = [np.array(0.44)]*self.n_arms
            return_times = np.array(1.0)

            for t in range(0,self.T):

                pulled_price = ts_learner.pull_arm() 
                price_value = ts_learner.expected_value(pulled_price)

                price = self.prices[pulled_price] #prezzo pullato

                mean_returns = np.mean(return_times)

                # il price value va moltiplicato per il numero di ritorni
                # expected prezzo arm estratto, moltiplicato per il numero medio di ritorni
                price_value *= mean_returns
                
                for bid in range(self.n_arms):  
                    gpts_learner.exp_cost[bid] = np.mean(past_costs[bid])
                    gpts_learner.upper_bound_cost[bid] = np.quantile(past_costs[bid], 0.8)
                
                pulled_bid = gpts_learner.pull_arm(price_value)

                news, costs = Benv.round(pulled_bid)
                news = int(np.ceil(news))

                # numero di persone che comprano
                buyer = Penv.round(pulled_price, news)

                # simuliamo il numero di volte in cui ogni cliente ritorna
                new_returns = np.ones(news) #self.cm.return_probability(lam = self.ret, size = news)
                
                # il price è moltiplicato per il numero medio di volte in cui gli utenti sono tornati
                reward = buyer*price*np.mean(new_returns) - np.sum(costs)
               
                # salviamo i nuovi costi ottenuti nei costi passati
                past_costs[pulled_bid] = np.append(past_costs[pulled_bid], costs)
                
                ts_learner.update_more(pulled_price, buyer, news-buyer) 
                gpts_learner.update(pulled_bid, news)

                rewards_this.append(reward)

                return_times = np.append(return_times, new_returns)
            
            self.rewards_full.append(rewards_this)

    def plot_reward(self):
        # TODO: plot reward
        pass

    def plot(self):

        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g', label="GPTS")
        plt.plot(np.quantile(np.cumsum(self.opt - self.rewards_full, axis=1), q=0.025, axis = 0),'g',linestyle='dashed', label="GPTS Confidence Interval 95%")
        plt.plot(np.quantile(np.cumsum(self.opt - self.rewards_full, axis=1), q=0.975,  axis = 0),'g',linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_6.png")
