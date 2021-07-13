from typing import Sized
from pricing import PersonGenerator
import numpy as np
import matplotlib.pyplot as plt
from environment import SpecificEnvironment, Environment
from learners import TS_Learner, Greedy_Learner
from tqdm import tqdm
from configmanager import *
import logging

from context import Context, ContextGenerator

class Experiment4():

    def __init__(self):
        self.cm = ConfigManager()
        self.pg = PersonGenerator(self.cm.get_classes(), self.cm.class_distribution)

        self.features = self.cm.get_features()

        self.bid = 0.42
        self.prices = self.cm.prices

        # probabilities (conv rates)
        self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06]
        #self.p = cm.aggr_conv_rates()

        self.n_arms = len(self.prices)
        self.opt = np.max(np.multiply(self.p, self.prices)) 

        self.T = 5000 # number of days
        self.n_experiments = 10

        self.reward_log = []
        self.reward_per_experiments = []    

        self.ts_reward_per_experiments = []
        self.gr_reward_per_experiments = []

 
            
    
    def run(self):
        for e in tqdm(range(0, self.n_experiments)):
            env = SpecificEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            env_normal = Environment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            gr_learner = Greedy_Learner(n_arms=self.n_arms)
            context_gen = ContextGenerator(10, self.cm.get_classes(), self.cm.features, self.prices)


            rewards = np.array([])
            rewards_ts = np.array([])
            rewards_gr = np.array([])

            for t in range(0,self.T): # 1 round is one day
                logging.debug(f'Experiment4.run() -> step {t} / {self.T}')

                # perform the offline context splitting, if needed
                if t>20:
                    context_gen.generate() 
                #context_gen.generate() 

                num_people = 20
                daily_reward = 0
                daily_reward_ts = 0
                daily_reward_gr = 0

                for _ in range(num_people): # p is a class e.g. ["Y", "I"], usually called user_class
                    
                    p = self.pg.generate_person()
                    # TS 
                    pulled_arm = context_gen.pull_arm(p)
                    reward = env.round(pulled_arm, p)

                    new_obs = [p, pulled_arm, reward]

                    context_gen.update(new_obs)

                    arm_exp_value = context_gen.expected_value_arm(p, pulled_arm)
                    daily_reward += reward
                    
                    # TS
                    #pulled_arm_ts = ts_learner.pull_arm()
                    #reward_ts = env_normal.round(pulled_arm_ts)
                    #ts_learner.update(pulled_arm_ts, reward)
                    #daily_reward_ts += reward_ts

                    # GR
                    #pulled_arm_gr = gr_learner.pull_arm()
                    #reward_gr = env_normal.round(pulled_arm_gr)
                    #gr_learner.update(pulled_arm_gr, reward)
                    #daily_reward_gr += reward_gr


                rewards = np.append(rewards, daily_reward)
                #rewards_ts = np.append(daily_reward_ts)
                #rewards_gr = np.append(daily_reward_gr)
                
            
            #    self.reward_log['t'] = {}
            #    self.reward_log['person'] = p
            #    self.reward_log['pulled_arm'] = pulled_arm
            #    self.reward_log['reward'] = pulled_arm
            #self.reward_per_experiments['exp'] = {}
            #self.reward_per_experiments['exp'] = self.reward_log
            self.reward_per_experiments.append(rewards)
            #self.ts_reward_per_experiments.append(ts_learner.collected_rewards)
            #self.gr_reward_per_experiments.append(ts_learner.collected_rewards)
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(np.cumsum(np.mean(self.opt*20 - self.reward_per_experiments, axis=0)), label='Context',color='b')
        #ax.plot(np.cumsum(np.mean(self.opt*20 - self.ts_reward_per_experiments, axis=0)), label='TS', color='y')
        #ax.plot(np.cumsum(np.mean(self.opt*20 - self.gr_reward_per_experiments, axis=0)), label='Greedy', color='r')
        ax.set_xlabel("t", fontsize=15)
        ax.set_ylabel("Regret", fontsize=18)

        ax.legend(loc="best")
        ax.margins(0.1)
        fig.tight_layout()
        
        plt.savefig("img/experiments/experiment_4.png")
        #plt.show()
