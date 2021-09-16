import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2, PriEnv
from learners import *
from scipy.stats import norm, beta

def generate_lambda(classes):
    total_prob = 0
    lam = 0
    for c in classes:
        total_prob += ConfigManager().class_distribution[c]
        lam += (ConfigManager().ret[c])*ConfigManager().class_distribution[c]
    return lam / total_prob  
    

    


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

        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 100 # number of days
        self.n_experiments = 5

    
    def run(self):

        self.rewards_full01 = []
        self.rewards_full23 = []

        Penv_01 = PriEnv(n_arms=self.n_arms, candidates=self.prices, classes = [0,1])            
        Penv_23 = PriEnv(n_arms=self.n_arms, candidates=self.prices, classes = [2,3])

        self.lam01 = generate_lambda(classes=[0,1])
        self.lam23 = generate_lambda(classes=[2,3])

        self.pricing_opt01,_ = Penv_01.compute_optimum()
        self.pricing_opt23,_ = Penv_23.compute_optimum()

        Benv_01 = BidEnv2(self.bids, self.num_people, classes=[0,1])
        Benv_23 = BidEnv2(self.bids, self.num_people, classes=[2,3])

        self.opt01 = Benv_01.compute_optimum(self.pricing_opt01, self.lam01)[0]
        self.opt23 = Benv_23.compute_optimum(self.pricing_opt23, self.lam23)[0]

        for e in tqdm(range(0, self.n_experiments)):

            ts_learner01 = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            ts_learner23 = TS_Learner(n_arms=self.n_arms, candidates=self.prices)

            gpts_learner01 = GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2)
            gpts_learner23 = GPTS2(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this01 = []
            rewards_this23 = []

            past_costs01 = [np.array(0.44)]*self.n_arms
            past_costs23 = [np.array(0.44)]*self.n_arms

            return_times01 = np.ones(1)
            return_times23 = np.ones(1)

            for t in range(0,self.T):
                 
                pulled_price01, price_value01 = ts_learner01.pull_arm()
                pulled_price23, price_value23 = ts_learner23.pull_arm()

                price01 = self.prices[pulled_price01]
                price23 = self.prices[pulled_price23]

                mean_returns01 = np.mean(return_times01)
                mean_returns23 = np.mean(return_times23)

                price_value01 *= mean_returns01
                price_value23 *= mean_returns23
                # il price value va moltiplicato per il numero di ritorni

                # update the number of cost per click in order to predict how much we expect to pay
                for bid in range(self.n_arms):  
                    gpts_learner01.exp_cost[bid] = np.quantile(past_costs01[bid], 0.8)
                    gpts_learner23.exp_cost[bid] = np.quantile(past_costs23[bid], 0.8)
                
                pulled_bid01 = gpts_learner01.pull_arm(price_value01)
                pulled_bid23 = gpts_learner23.pull_arm(price_value23)

                news01, costs01 = Benv_01.round(pulled_bid01)
                news23, costs23 = Benv_23.round(pulled_bid23)

                news01 = int(news01+0.5)
                news23 = int(news23+0.5)

                buyer01 = Penv_01.round(pulled_price01, news01)
                buyer23 = Penv_23.round(pulled_price23, news23)

                new_returns01 = 1+np.random.poisson(lam = self.lam01, size = news01)
                new_returns23 = 1+np.random.poisson(lam = self.lam23, size = news23)
                # simuliamo il numero di volte in cui ogni cliente ritorna

                pricing_rew01 = buyer01*price01-np.sum(costs01)
                pricing_rew23 = buyer01*price23-np.sum(costs23)

                reward01 = news01*self.pricing_opt01-np.sum(costs01)
                reward23 = news23*self.pricing_opt23-np.sum(costs23)
                # il price è moltiplicato per il numero medio di volte in cui gli utenti sono tornati

                not_buyer01 = news01-buyer01
                not_buyer23 = news23-buyer23

                past_costs01[pulled_bid01] = np.append(past_costs01[pulled_bid01], costs01)
                past_costs23[pulled_bid23] = np.append(past_costs23[pulled_bid23], costs23)
                
                ts_learner01.update_more(pulled_price01, pricing_rew01, buyer01, not_buyer01)
                ts_learner23.update_more(pulled_price23, pricing_rew23, buyer23, not_buyer23)
                gpts_learner01.update(pulled_bid01, news01)
                gpts_learner23.update(pulled_bid23, news23)

                rewards_this01.append(reward01)
                rewards_this23.append(reward23)

                return_times01 = np.append(return_times01, new_returns01)
                return_times23 = np.append(return_times23, new_returns23)
            
            self.rewards_full01.append(rewards_this01)
            self.rewards_full23.append(rewards_this23)


    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt01 - self.rewards_full01, axis = 0)),'g', label="macro01")
        plt.plot(np.cumsum(np.mean(self.opt23 - self.rewards_full23, axis = 0)),'g', label="macro23")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_7_new.png")

