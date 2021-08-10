import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from learners import TS_Learner, Greedy_Learner, UCB1
from tqdm import tqdm
from configmanager import *


class Experiment3():

    def __init__(self):
        cm = ConfigManager()

        self.prices = cm.prices # candidates

        self.p = cm.aggr_conv_rates()
        self.n_arms = cm.n_arms
        self.opt = np.max(np.multiply(self.p, self.prices)) 

        self.T = 365 # number of days
        self.n_experiments = 100

        self.colors = cm.colors

        self.gr_reward_per_experiments = []
        self.uc_reward_per_experiments = []
        self.ts_reward_per_experiments = []

        np.random.seed(123)
    
    def run(self):

        for e in tqdm(range(0, self.n_experiments)):
            env = Environment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)

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
        plt.plot(np.cumsum(np.mean(self.opt - self.gr_reward_per_experiments, axis=0)), self.colors[0], label="Greedy")
        plt.plot(np.cumsum(np.mean(self.opt - self.uc_reward_per_experiments, axis=0)), self.colors[1], label="UCB1")
        plt.plot(np.cumsum(np.mean(self.opt - self.ts_reward_per_experiments, axis=0)), self.colors[3], label="TS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_3.png")