from pricing import PersonGenerator
import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm

from environment import PricingEnvironment, BiddingEnvironment, SpecificEnvironment, BidEnv2
from context import ContextGenerator
from learners import *
from scipy.stats import norm, beta



class Experiment7():
    
    def __init__(self):

        self.cm = ConfigManager()
        self.colors = self.cm.colors

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.num_people = 10000*np.array(self.cm.class_distribution)

        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        
        # bidding 
        self.bids = np.array(self.cm.bids)

        self.T = 200 # number of days
        self.n_experiments = 5


    def run(self):

        Penvs = []
        Benvs = []

        for c in range(4):
            Penvs.append(PricingEnvironment(self.n_arms, self.cm.conv_rates[c] ,self.prices))
            Benvs.append(BiddingEnvironment(self.bids, self.cm.new_clicks_function_mean(self.bids, c, self.num_people[c]), self.cm.new_clicks_function_mean(self.bids, c, self.num_people[c]), classe = c))

        ### compute pricing and bidding optima
        self.pricing_opt = np.zeros(4)
        self.opt = np.zeros(4)
        for c in range(4):
            self.pricing_opt[c] = Penvs[c].get_pricing_optimum()
            self.opt[c] = Benvs[c].get_optimum(self.pricing_opt[c], self.cm.ret[c])

        self.rewards_full = [[], [], [], []]

        
        for e in tqdm(range(0, self.n_experiments)):

            ### Define the learners

            ts_learners = []
            gpts_learners = []           
            for c in range(4):
                ts_learners.append(TS_Learner(n_arms=self.n_arms, candidates=self.prices))
                gpts_learners.append(GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2))

            ### Define the lists of pasts costs per click
            

            past_costs = []
            for c in range(4):
                past_costs.append([np.array(0.44)]*self.n_arms)

            ### Define the list of past return times
           

            return_times = []
            for c in range(4):
                return_times.append(np.ones(1))

            ### rewards


            rewards_this = [[], [], [], []]

            ### Real experiment starts


            for t in range(0,self.T):

                for c in range(4):
                    pulled_price, price_value = ts_learners[c].pull_arm()
                    price = self.prices[pulled_price]

                    mean_returns = np.mean(return_times[c])
                    price_value *= mean_returns
                    # il price value va moltiplicato per il numero di ritorni

                    for bid in range(self.n_arms):  
                        gpts_learners[c].exp_cost[bid] = np.quantile(past_costs[c][bid], 0.8)
                    
                    pulled_bid = gpts_learners[c].pull_arm(price_value)

                    news, costs = Benvs[c].round(pulled_bid)
                    news = int(news+0.5)
                    buyer = Penvs[c].round(pulled_price, news)

                    new_returns = 1+np.random.poisson(lam = self.cm.ret[c], size = news)
                    # simuliamo il numero di volte in cui ogni cliente ritorna

                    pricing_rew = buyer*price-np.sum(costs)
                    reward = buyer*price*np.mean(new_returns)-np.sum(costs)
                    # il price è moltiplicato per il numero medio di volte in cui gli utenti sono tornati

                    not_buyer = news-buyer

                    past_costs[c][pulled_bid] = np.append(past_costs[c][pulled_bid], costs)
                    
                    ts_learners[c].update_more(pulled_price, pricing_rew, buyer, not_buyer)
                    gpts_learners[c].update(pulled_bid, news)

                    rewards_this[c].append(reward)

                    return_times = np.append(return_times, new_returns)

            for c in range(4):
                self.rewards_full[c].append(rewards_this[c])
        

        '''
        for c in range(4):
            print(Penvs[c].candidates)
            print(Benvs[c].means)
            print(self.pricing_opt[c])
            print(self.opt[c])
        '''

        
        

    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        for c in range(4):
            plt.plot(np.cumsum(np.mean(self.opt[c] - self.rewards_full[c], axis = 0)), color = self.colors[c], label=str(c))
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_7.png")

        #plt.show()