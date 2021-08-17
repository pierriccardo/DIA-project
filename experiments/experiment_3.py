import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

from pricing import PersonGenerator
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
        self.n_experiments = 10

        self.colors = cm.colors

        self.gr_reward_per_experiments = []
        self.uc_reward_per_experiments = []
        self.ts_reward_per_experiments = []

        self.gr_regret_per_experiment = []
        self.uc_regret_per_experiment = []
        self.ts_regret_per_experiment = []

        self.opt_per_experiment = []
    
    def run(self):
        
        pg = PersonGenerator()

        for e in tqdm(range(0, self.n_experiments)):
            env = Environment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)

            gr_learner = Greedy_Learner(n_arms=self.n_arms)
            uc_learner = UCB1(n_arms=self.n_arms, prices=self.prices)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)

            gr_regret = []
            uc_regret = []
            ts_regret = []

            for t in range(0,self.T):

                gr_daily_reward = 0
                uc_daily_reward = 0
                ts_daily_reward = 0

                people = 100
                daily_opt = self.opt * people

                gr_pulled_arm = gr_learner.pull_arm()
                uc_pulled_arm = uc_learner.pull_arm()
                ts_pulled_arm = ts_learner.pull_arm()
                
                for _ in range(people):
                    gr_reward = env.round(gr_pulled_arm)
                    uc_reward = env.round(uc_pulled_arm)
                    ts_reward = env.round(ts_pulled_arm)   

                    gr_daily_reward += gr_reward
                    uc_daily_reward += uc_reward
                    ts_daily_reward += ts_reward
                
                    gr_learner.update(gr_pulled_arm, gr_reward)
                    uc_learner.update(uc_pulled_arm, uc_reward)
                    ts_learner.update(ts_pulled_arm, ts_reward) 

                gr_regret.append(daily_opt - gr_daily_reward)
                uc_regret.append(daily_opt - uc_daily_reward)
                ts_regret.append(daily_opt - ts_daily_reward)

            self.gr_regret_per_experiment.append(gr_regret)
            self.uc_regret_per_experiment.append(uc_regret)
            self.ts_regret_per_experiment.append(ts_regret)      

    def plot(self):

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.gr_regret_per_experiment, axis=0)), self.colors[0], label="Greedy")
        plt.plot(np.cumsum(np.mean(self.uc_regret_per_experiment, axis=0)), self.colors[1], label="UCB1")
        plt.plot(np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)), self.colors[3], label="TS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_3.png")