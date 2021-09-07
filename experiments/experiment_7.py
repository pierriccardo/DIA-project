from pricing import PersonGenerator
import numpy as np
import matplotlib.pyplot as plt

from configmanager import ConfigManager
from tqdm import tqdm

from environment import PricingEnvironment, BiddingEnvironment, SpecificEnvironment
from context import ContextGenerator
from learners import *
from scipy.stats import norm, beta



class Experiment7():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = np.array(self.cm.prices) # candidates

        self.p = self.cm.aggr_conv_rates()
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()

        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        
        # bidding 
        self.bids = np.array(self.cm.bids)
        self.sigma = 10
        
        # self.means = self.cm.new_clicks(self.bids)
        self.means = self.cm.aggregated_new_clicks_function_mean(self.bids, self.num_people)
        self.sigmas = self.cm.aggregated_new_clicks_function_sigma(self.bids, self.num_people)

        self.opt = np.max(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids)))
        indice = np.argmax(self.means * (self.opt_pricing - self.cm.mean_cc(self.bids)))

        print(self.means[indice])
        print(self.opt_pricing)
        print(self.cm.mean_cc(self.bids)[indice])
        
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 200 # number of days
        self.n_experiments = 8

    def run(self):

        self.rewards_full = []
        pg = PersonGenerator()

        for e in tqdm(range(0, self.n_experiments)):

            env = SpecificEnvironment(n_arms=self.n_arms, candidates=self.prices)
            penv = PricingEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            benv = BiddingEnvironment(self.bids, self.means, self.sigmas)
            
            
            ts_learners = []
            gpts_learners = []
            rewards_this = []
            for macros in range(3):
                ts_learners.append(TS_Learner(n_arms=self.n_arms, candidates=self.prices))
                gpts_learners.append(GPTS_learner_positive(n_arms=self.n_arms, arms=self.bids, threshold=0.2))
                rewards_this.append([])
            


            for t in range(0,self.T): # 1 round is one day

                
                num_people = pg.generate_people_num()

                for _ in range(num_people): # p is a class e.g. ["Y", "I"], usually called user_class
                    
                    p_class, p_labels = pg.generate_person()
                
                    pulled_arm, _ = context_gen.pull_arm(p_labels)
                    reward = env.round(pulled_arm, p_class)

                    current_opt = np.max(np.multiply(self.cm.conv_rates[p_class], self.prices))

                    new_obs = [p_labels, pulled_arm, reward]
                    context_gen.update(new_obs)

                    pulled_price, price_value = context_gen.pull_arm()
                    price = self.prices[pulled_price]

                    pulled_bid = gpts_learner.pull_arm(price_value)
                    
                    news, cost = benv.round(pulled_bid)
                    buyer = penv.round(pulled_price, news)

                    reward = buyer*price - cost

                    not_buyer = news - buyer
                    
                    ts_learner.update_more(pulled_price, reward, buyer, not_buyer)
                    gpts_learner.update(pulled_bid, news)

                    rewards_this.append(reward)
            
            self.rewards_full.append(rewards_this)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.rewards_full, axis = 0)),'g', label="GPTS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_7.png")

        #plt.show()

