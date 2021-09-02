import numpy as np
from configmanager import ConfigManager
from tqdm import tqdm
from environment import PricingEnvironment
from environment import BiddingEvironment
from learners import *
from scipy.stats import norm, beta
import matplotlib.pyplot as plt


class Experiment6_b():
    
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
        
        
        # self.means = self.cm.new_clicks(self.bids)
        self.means = self.cm.aggregated_new_clicks_function_mean(self.bids, self.num_people)
        self.sigmas = self.cm.aggregated_new_clicks_function_sigma(self.bids, self.num_people)


        # return prob #######################################################################################
        self.ret = self.cm.avg_ret

        self.opt = np.max(self.means * (self.opt_pricing*self.ret - self.cm.mean_cc(self.bids))) ## corretto l'ottimo per tener conto dei ritorni
        indice = np.argmax(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids)))

        print(self.means[indice])
        print(self.opt_pricing)
        print(self.cm.mean_cc(self.bids)[indice])
        
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 180 # number of days
        self.n_experiments = 10

    def run(self):
        self.rewards_full = []

        for e in tqdm(range(0, self.n_experiments)):

            Penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            Benv = BiddingEvironment(self.bids, self.means, self.sigmas)
            

            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            gpts_learner = GPTS_learner_positive(n_arms=self.n_arms, arms=self.bids, threshold=0.2)



            rewards_this = []

            return_times = np.ones(1)
            # qui collezioniamo i tempi di ritorno di ogni cliente

            ### mettiamo i round a vuoto dovuti al ritardo


            for t in range(0,self.T):
                 
                pulled_price, price_value = ts_learner.pull_arm()
                price = self.prices[pulled_price]

                mean_returns = np.mean(return_times)
                price_value *= mean_returns
                # il price value va moltiplicato per il numero di ritorni
                
                pulled_bid = gpts_learner.pull_arm(price_value)

                
                news, cost = Benv.round(pulled_bid)
                news = int(news)
                buyer = Penv.round(pulled_price, news)

                new_returns = np.random.poisson(lam = self.ret, size = news)
                # simuliamo il numero di volte in cui ogni cliente ritorna

                pricing_rew = buyer*price-cost
                reward = buyer*price*np.mean(new_returns)-cost
                # il price è moltiplicato per il numero medio di volte in cui gli utenti sono tornati

                not_buyer = news-buyer
                
                # RITARDO: per i primi 30 round non aggiorniamo nulla
                if (t > 30):
                    ts_learner.update_more(pulled_price, pricing_rew, buyer, not_buyer)
                    gpts_learner.update(pulled_bid, news)

                    return_times = np.append(return_times, new_returns)
                
                rewards_this.append(reward)
            
            self.rewards_full.append(rewards_this)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        
        
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g')

        ## dove inizia l'apprendimento?
        plt.axvline(x=30, ymin=0, ymax=10000., color='r', linestyle=':', linewidth=3)

        plt.legend(["GPTS"])
        plt.savefig("img/experiments/experiment_6_return_time.png")
        

        #plt.show()