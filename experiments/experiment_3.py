import numpy as np
from configmanager import ConfigManager
from tqdm import tqdm
from environment import PricingEnvironment
from environment import BiddingEvironment
from learners import *
from scipy.stats import norm, beta
import matplotlib.pyplot as plt


class Experiment3():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates
        self.news = 1000  # utenti che clickano

        # self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        self.best_arm = np.argmax(np.multiply(self.p, self.prices))
       
        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 360 # number of days
        self.n_experiments = 100

    def run(self):

        self.rewards_full_greedy = []
        self.rewards_full_UCB1 = []
        self.rewards_full_TS = []

        for e in tqdm(range(0, self.n_experiments)):

            
            
            Greedy_learner = Greedy_Learner(n_arms=self.n_arms)
            UCB1_learner = UCB1(n_arms=self.n_arms, prices=self.prices, alpha=1.)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)

            Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            rewards_this_greedy = []

            for t in range(0,self.T):
                 
                pulled_price = Greedy_learner.pull_arm()
                price = self.prices[pulled_price]                

                buyer = Penv.round(pulled_price, self.news)

                reward = buyer*price
                
                not_buyer = self.news - buyer
                
                Greedy_learner.update(pulled_price, reward)

                rewards_this_greedy.append(reward)
            
            self.rewards_full_greedy.append(rewards_this_greedy)


            Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            rewards_this_UCB1 = []

            for t in range(0,self.T):
                 
                pulled_price = UCB1_learner.pull_arm()
                price = self.prices[pulled_price]                

                buyer = Penv.round(pulled_price, self.news)

                reward = buyer*price
                '''
                if (t % 180 == 179):
                    print(t)
                    print(UCB1_learner.empirical_means)
                    print('braccio tirato: '+str(pulled_price)+' braccio ottimo: '+str(self.best_arm))
                    #print('empirical opt pricing: '+str(buyer*price/self.news)+' theoretical opt pricing: '+str(self.opt_pricing))
                '''
                not_buyer = self.news - buyer
                
                UCB1_learner.update_more(pulled_price, reward, buyer, not_buyer)

                rewards_this_UCB1.append(reward)
            
            self.rewards_full_UCB1.append(rewards_this_UCB1)

            ## reset environment
            Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            rewards_this_TS = []
            ## Ciclo per TS
            for t in range(0,self.T):
                 
                pulled_price, _ = ts_learner.pull_arm()
                price = self.prices[pulled_price]

                # problema: il braccio tirato dal learner di bidding
                # deve dipendere da price
                

                buyer = Penv.round(pulled_price, self.news)

                reward = buyer*price

                #if (t % 199 == 0):
                #   print('empirical opt pricing: '+str(buyer*price/self.news)+' theoretical opt pricing: '+str(self.opt_pricing))
                
                not_buyer = self.news - buyer
                
                ts_learner.update_more(pulled_price, reward, buyer, not_buyer)

                rewards_this_TS.append(reward)
            
            self.rewards_full_TS.append(rewards_this_TS)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        
        
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        
        plt.plot(np.cumsum(np.mean(self.opt_pricing*self.news - self.rewards_full_greedy, axis = 0)),'r' , label="Greedy")
        plt.plot(np.cumsum(np.mean(self.opt_pricing*self.news - self.rewards_full_UCB1, axis = 0)), 'g' , label="UCB1")
        plt.plot(np.cumsum(np.mean(self.opt_pricing*self.news - self.rewards_full_TS, axis = 0)), 'b', label="TS")
        
        plt.savefig("img/experiments/experiment_3.png")
        

        #plt.show()