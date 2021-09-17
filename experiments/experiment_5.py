import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape, size

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2 
from learners import *
from scipy.stats import norm, beta
import logging


class Experiment5():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.num_people = self.cm.num_people*np.array(self.cm.class_distribution)

        self.DELAY = 0

        # self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        
        # bidding 
        self.bids = np.array(self.cm.bids)      

        self.T = 200 # number of days
        self.n_experiments = 10

        self.reward_per_experiment = []

        

    def run(self):
        Benv = BidEnv2(self.bids, self.num_people)

        # opt è il valore della bid ottima
        self.opt = Benv.compute_optimum(self.opt_pricing)[0]

        for e in tqdm(range(0, self.n_experiments)):

            pull_arm_buffer = []
            news_buffer = []

            gpts_learner = GPTS(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = []

            # i costi delle bid selezionate in passato
            # [ [0.44], [0.44], ....]
            past_costs = [np.array([0.44])]*self.n_arms
            
            for t in range(0,self.T):
                if t>self.DELAY:

                    for bid in range(self.n_arms):  # update quantiles of expected costs
                        # mettiamo in fila i costi passati e prendiamo quello corrispondente all'80
                        # dal costo più alto prendiamo quello che si classifica 20esimo su 100 
                        
                        gpts_learner.upper_bound_cost[bid] = np.quantile(past_costs[bid], 0.8)
                        # media dei costi passati
                        gpts_learner.exp_cost[bid] = np.mean(past_costs[bid])

                # dato che il prezzo è fissato, passiamo al gpts
                # il valore pricing ottimo
                pulled_bid = gpts_learner.pull_arm(self.opt_pricing)
                
                # news = new clicks
                news, costs = Benv.round(pulled_bid)
                
                reward = news*self.opt_pricing - np.sum(costs)

                logging.info('exp5() -> reward: '+str(reward)+' ottimo teorico: '+str(self.opt))

                # aggiorniamo i past cost aggiungendo i nuovi costs ottenuti
                past_costs[pulled_bid] = np.append(past_costs[pulled_bid], costs)

                news_buffer.append(news)
                pull_arm_buffer.append(pulled_bid)
                
                if t>self.DELAY+1:
                    gpts_learner.update(pull_arm_buffer[-self.DELAY], news_buffer[-self.DELAY])

                rewards_this.append(reward)
            
            self.reward_per_experiment.append(rewards_this)

    def plot_reward(self):
        plt.figure(1)
        plt.ylabel('Reward')
        plt.xlabel('t')

        plt.plot(np.mean(self.reward_per_experiment, axis = 0),'g', label="GPTS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        name = "img/experiments/experiment_5_reward_delay"
        name += str(self.DELAY)
        name += str('.png')
        plt.savefig(name)

    def plot_regret(self):
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.reward_per_experiment, axis = 0)),color=self.cm.colors[0], label="GPTS")
        plt.plot(np.quantile(np.cumsum(self.opt - self.reward_per_experiment, axis=1), q=0.025, axis = 0),'g',linestyle='dashed', label="GPTS Confidence Interval 95%")
        plt.plot(np.quantile(np.cumsum(self.opt - self.reward_per_experiment, axis=1), q=0.975,  axis = 0),'g',linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        name = "img/experiments/experiment_5_regret_delay"
        name += str(self.DELAY)
        name += str('.png')
        plt.savefig(name)

    def plot(self):
        self.plot_reward()
        self.plot_regret()