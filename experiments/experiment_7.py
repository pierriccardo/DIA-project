import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin

from configmanager import ConfigManager
from tqdm import tqdm
from environment import BidEnv2, PricingEnvironment, BiddingEnvironment
from learners import *
from scipy.stats import norm, beta



class Experiment7():

    NAME = 'Experiment 7'
    
    def __init__(self, days=365, n_exp=10, delay=30):
        self.cm = ConfigManager()

        # pricing
        self.prices = self.cm.prices # candidates
        self.n_arms = len(self.prices)
        
        # bidding 
        self.bids = np.array(self.cm.bids)

        self.T = days # number of days
        self.n_experiments = n_exp
        self.DELAY = delay

        self.results = {}

    def run(self):
        self.run_exp([0,1],[2,3])
        self.run_exp([2],[0,1,3])
        self.run_exp([3],[0,1,2])

    def run_exp(self, classes, not_classes):# self.classes

        self.classes = classes
        self.not_classes = not_classes

        self.num_people = self.cm.num_people*np.array(self.cm.class_distribution)
        for c in self.not_classes:
            self.num_people[c] = 0


        self.rewards_full = [] #lista delle reward dei vari esperimenti 
        self.regret_full = []

        self.p = self.cm.aggr_conv_rates(self.classes)
        # calcola i ritorni medi delle varie classi
        self.ret = self.cm.aggr_return_probability(self.classes)

        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        self.best_price = np.argmax(np.multiply(self.p, self.prices))


        #### questo è quello che c'era veramente in run

        Benv = BidEnv2(self.bids, self.num_people)
        Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
        self.opt, self.best_bid = Benv.compute_optimum(self.opt_pricing, self.ret)

        for e in tqdm(range(0, self.n_experiments)):

            pull_arm_buffer = []
            news_buffer = []
            
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            gpts_learner = GPTS(n_arms=self.n_arms, arms=self.bids, threshold=0.2)

            rewards_this = [] # reward dell'esperimento 
            regret_this = []

            past_costs = [np.array(0.44)]*self.n_arms
            
            return_times = np.array(1.0)

            for t in range(0,self.T):

                pulled_price = self.best_price#ts_learner.pull_arm() 
                price_value = ts_learner.expected_value(pulled_price)

                price = self.prices[pulled_price] #prezzo pullato

                mean_returns = np.mean(return_times)

                # il price value va moltiplicato per il numero di ritorni
                # expected prezzo arm estratto, moltiplicato per il numero medio di ritorni
                price_value *= mean_returns

                if t>self.DELAY:
                    for bid in range(self.n_arms):  
                        gpts_learner.exp_cost[bid] = np.mean(past_costs[bid])
                        gpts_learner.upper_bound_cost[bid] = np.quantile(past_costs[bid], 0.8)
                
                pulled_bid = gpts_learner.pull_arm(price_value)

                news, costs = Benv.round(pulled_bid)
                pseudo_news, pseudo_costs = Benv.round(self.best_bid)
                news = int(np.ceil(news))
                pseudo_news = int(np.ceil(pseudo_news))

                # numero di persone che comprano
                buyer = Penv.round(pulled_price, news)

                new_returns = 1+self.cm.return_probability(lam = self.ret, size = news)
                # simuliamo il numero di volte in cui ogni cliente ritorna

                # il price è moltiplicato per il numero medio di volte in cui gli utenti sono tornati
                pseudo_reward = pseudo_news*self.opt_pricing*(self.ret+1) - np.sum(pseudo_costs)
                reward = buyer*price*np.mean(new_returns) - np.sum(costs)

               
                # salviamo i nuovi costi ottenuti nei costi passati
                past_costs[pulled_bid] = np.append(past_costs[pulled_bid], costs)
                
                ts_learner.update_more(pulled_price, buyer, news-buyer) 
                
                news_buffer.append(news)
                pull_arm_buffer.append(pulled_bid)
                
                if t>self.DELAY:
                    gpts_learner.update(pull_arm_buffer[-(self.DELAY+1)], news_buffer[-(self.DELAY+1)])
                

                rewards_this.append(reward)
                regret_this.append(pseudo_reward - reward)

                if t>self.DELAY:
                    return_times = np.append(return_times, new_returns)
            
            self.rewards_full.append(rewards_this)
            self.regret_full.append(regret_this)
            #print(past_costs[0])

        self.results[classes[0]] = {}
        self.results[classes[0]]['regret'] = self.regret_full
        self.results[classes[0]]['reward'] = self.rewards_full
        self.results[classes[0]]['opt'] = self.opt

    def plot(self):
        self.plot_reward()
        self.plot_regret()

    def plot_regret(self):
        
        plt.figure(71)
        plt.ylabel('Pseudo Regret')
        plt.xlabel('t')
        classes = [0,2,3]
        for c in classes:
            regret_full = self.results[c]['regret']
            plt.plot(np.cumsum(np.mean(regret_full, axis = 0)),color=self.cm.colors[c], label=self.cm.class_labels[c])
            plt.plot(np.quantile(np.cumsum(regret_full, axis=1), q=0.025, axis = 0), self.cm.colors[c],linestyle='dashed', label="GPTS Confidence Interval 95%")
            plt.plot(np.quantile(np.cumsum(regret_full, axis=1), q=0.975,  axis = 0), self.cm.colors[c],linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig(f"img/experiments/experiment_7_regret_delay_{self.DELAY}.png")


    def plot_reward(self):
        plt.figure(72)
        plt.ylabel('Reward')
        plt.xlabel('t')
        classes = [0,2,3]
        for c in classes:
            reward_full = self.results[c]['reward']
            opt = self.results[c]['opt']
            plt.plot(np.mean(reward_full, axis = 0),color=self.cm.colors[c], label=self.cm.class_labels[c])
            plt.plot([opt for i in range(self.T)], color=self.cm.colors[c],linestyle='dashed', label= "opt" + self.cm.class_labels[c])
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))

        plt.savefig(f"img/experiments/experiment_7_reward_delay_{self.DELAY}.png")
        
