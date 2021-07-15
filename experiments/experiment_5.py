from learners import GPTS_learner_positive
from environment import BiddingEvironment
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from configmanager import *

class Experiment5():

    def __init__(self):

        cm = ConfigManager()

        self.n_arms = 10
        self.bids = np.array(cm.bids)
        self.prices = cm.prices
        self.num_people = 100000*np.array(cm.class_distribution)
        self.p = cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        self.means = cm.aggregated_new_clicks_function_mean(self.bids, self.num_people)
        self.sigmas = cm.aggregated_new_clicks_function_sigma(self.bids, self.num_people)
        self.opt = np.max(self.means * (self.opt_pricing - cm.cc(self.bids)))
        self.optimal_bid = np.argmax(self.means * (self.opt_pricing - cm.cc(self.bids)))
        self.T = 300
        self.n_experiments = 5
        self.gpts_reward_per_experiment = []
        self.p_arms = []

    def run(self):
        for e in tqdm(range(self.n_experiments)):
            '''
            print('num_people '+str(self.num_people))
            print('medie '+str(self.means))
            print('sd '+str(self.sigmas))
            print('ottimo di pricing '+str(self.opt_pricing))
            print('ottimo totale '+str(self.opt))
            '''
            env = BiddingEvironment(self.bids, self.means, self.sigmas)
            gpts_learner = GPTS_learner_positive(n_arms=self.n_arms, arms=self.bids, threshold=0.2) 
            # qui metto anche bid perchè per implementare GP serve sapere le distanze tra i dati

            for t in range(self.T):

                pulled_arm = gpts_learner.pull_arm()
                reward = env.round(pulled_arm, self.opt_pricing)
                gpts_learner.update(pulled_arm, reward)
                self.p_arms.append(pulled_arm)

            self.gpts_reward_per_experiment.append(gpts_learner.collected_rewards)
        print(self.optimal_bid)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.figure(0)
        plt.hist(self.p_arms)
        plt.savefig("img/experiments/experiment_5_hist.png")
        
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.gpts_reward_per_experiment, axis = 0)),'g')
        plt.legend(["GPTS"])
        plt.savefig("img/experiments/experiment_5.png")
        

        #plt.show()