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
        self.reward_per_experiments = []

        self.contexts_per_experiments = []

        
    
    def run(self):
        for e in tqdm(range(0, self.n_experiments)):
            env = SpecificEnvironment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            context_gen = ContextGenerator(10, self.cm.get_classes(), self.cm.features, self.prices, self.obs)
        
            for t in range(0,self.T): # 1 round is one day

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
            self.reward_per_experiments.append(self.reward_log)
            self.contexts_per_experiments.append(context_gen.contexts)
        
  
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = ['r', 'g', 'b', 'y']
        contexts = self.contexts_per_experiments[0]
        
        for c in contexts:
            context_aggr_conv_rates = self.cm.class_aggr_conv_rates(c.classes)
            opt = np.max(np.multiply(context_aggr_conv_rates, self.prices))
            print(f'opt for class {c.classes} is {opt}')
            regret_list = []
            for exp in self.reward_per_experiments:
                for log in exp:
                    for ctx_c in c.classes:
                        if log[0][1] in ctx_c and log[0][1] in ctx_c:
                            regret_list.append(opt - log[3]) # log[3] is the arm_exp_value 
                            

            color = colors[np.random.choice(len(colors))]
            colors.remove(color)
            ax.plot(np.cumsum(np.mean(regret_list, axis=0)), label=f'Context {c.id}', color=color)
        ax.set_xlabel("t", fontsize=15)
        ax.set_ylabel("Regret", fontsize=18)

        ax.legend(loc="best")
        ax.margins(0.1)
        fig.tight_layout()
        
        plt.savefig("img/experiments/experiment_4.png")
        #plt.show()
