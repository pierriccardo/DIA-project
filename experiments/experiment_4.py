from pricing import PersonGenerator
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment, SpecificEnvironment
from learners import TS_Learner
from tqdm import tqdm
from configmanager import *
import logging

from context import Context, ContextGenerator

class Experiment4():

    def __init__(self):
        self.cm = ConfigManager()
        self.pg = PersonGenerator(self.cm.get_classes(), self.cm.class_distribution)

        self.bid = 0.42
        self.prices = self.cm.prices

        # probabilities (conv rates)
        self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06]
        #p = cm.aggr_conv_rates()

        self.n_arms = len(self.prices)
        self.opt = np.max(np.multiply(self.p, self.prices)) 

        self.T = 365 # number of days
        self.n_experiments = 1

        self.obs = []
        self.reward_log = []
    
    def run(self):
        for e in tqdm(range(0, self.n_experiments)):
            env = SpecificEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            context_gen = ContextGenerator(10, self.cm.get_classes(), self.cm.features, self.prices, self.obs)

            for t in tqdm(range(0,self.T)): # 1 round is one day

                # perform the offline context splitting, if needed
                context_gen.generate() 

                num_people = 1
                for _ in range(num_people): # p is a class e.g. ["Y", "I"], usually called user_class
                    
                    p = self.pg.generate_person()
                    # TS 
                    pulled_arm = context_gen.pull_arm(p)
                    reward = env.round(pulled_arm)

                    new_obs = [p, pulled_arm, reward]

                    context_gen.update(new_obs)

                    arm_exp_value = context_gen.expected_value_arm(p, pulled_arm)

                self.reward_log.append([p, pulled_arm, reward, arm_exp_value])
  
    
    def plot(self):
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.opt - self.ts_reward_per_experiments, axis=0)), 'r')
        plt.legend(["TS"])
        plt.savefig("img/experiments/experiment_4.png")
        #plt.show()