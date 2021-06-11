import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, SpecificEnvironment
from learners import TS_Learner, Greedy_Learner, UCB1
from tqdm import tqdm
from configmanager import *


class Experiment3():

    def __init__(self):
        self.cm = ConfigManager()

        self.prices = self.cm.prices # candidates

        self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        #p = cm.aggr_conv_rates()
        self.n_arms = len(self.prices)
        self.opt = np.max(np.multiply(self.p, self.prices)) 

        self.T = 365 # number of days
        self.n_experiments = 100

        self.gr_reward_per_experiments = []
        self.uc_reward_per_experiments = []
        self.ts_reward_per_experiments = []
    
    def run(self):

        for e in tqdm(range(0, self.n_experiments)):
            env = SpecificEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            gr_learner = Greedy_Learner(n_arms=self.n_arms)
            uc_learner = UCB1(n_arms=self.n_arms, prices=self.prices)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            for t in range(0,self.T):
                
                # Greedy Learner
                pulled_arm = gr_learner.pull_arm()
                reward = env.round(pulled_arm)
                gr_learner.update(pulled_arm, reward)

                # UCB1
                pulled_arm = uc_learner.pull_arm()
                reward = env.round(pulled_arm)
                uc_learner.update(pulled_arm, reward)

                # TS 
                pulled_arm = ts_learner.pull_arm()
                reward = env.round(pulled_arm)
                ts_learner.update(pulled_arm, reward)

            self.gr_reward_per_experiments.append(gr_learner.collected_rewards)
            self.uc_reward_per_experiments.append(uc_learner.collected_rewards)
            self.ts_reward_per_experiments.append(ts_learner.collected_rewards)
    
    def plot(self):

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.opt - self.ts_reward_per_experiments, axis=0)), 'r')
        plt.plot(np.cumsum(np.mean(self.opt - self.gr_reward_per_experiments, axis=0)), 'g')
        plt.plot(np.cumsum(np.mean(self.opt - self.uc_reward_per_experiments, axis=0)), 'y')
        plt.legend(["TS", "Greedy", "UCB1", "TS_prices"])
        plt.savefig("img/experiments/experiment_3.png")
        #plt.show()