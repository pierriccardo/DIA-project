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
        self.sigma = 10
        self.prices = cm.prices
        self.p = [.1, .03, .34, .28, .12, .19, .05, .56, .26, .35]# cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 
        self.means = cm.new_clicks(self.bids)
        self.opt = np.max(self.means * (self.opt_pricing - self.bids))
        self.T = 200
        self.n_experiments = 10
        self.gpts_reward_per_experiment = []
        self.p_arms = []

    def run(self):
        for e in tqdm(range(self.n_experiments)):
            env = BiddingEvironment(self.bids, self.means, self.sigma, self.opt_pricing)
            gpts_learner = GPTS_learner_positive(n_arms=self.n_arms, arms=self.bids, threshold=0.2) 
            # qui metto anche bid perchè per implementare GP serve sapere le distanze tra i dati

            for t in range(self.T):

                pulled_arm = gpts_learner.pull_arm()
                reward = env.round(pulled_arm)
                gpts_learner.update(pulled_arm, reward)
                self.p_arms.append(pulled_arm)

            self.gpts_reward_per_experiment.append(gpts_learner.collected_rewards)

    def plot(self):
        #sns.distplot(np.array(p_arms))

        plt.hist(self.p_arms)
        
        plt.figure(0)
        plt.ylabel('Regret')
        plt.xlabel('t')
        plt.plot(np.cumsum(np.mean(self.opt - self.gpts_reward_per_experiment, axis = 0)),'g')
        plt.legend(["GPTS"])
        plt.savefig("img/experiments/experiment_5.png")

        #plt.show()