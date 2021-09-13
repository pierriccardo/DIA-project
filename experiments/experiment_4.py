from typing import Sized
from pricing import PersonGenerator
import numpy as np
import matplotlib.pyplot as plt
from environment import SpecificEnvironment
from tqdm import tqdm
from configmanager import *
import logging

from context2 import ContextGenerator

class Experiment4():

    def __init__(self):
        self.cm = ConfigManager()
        
        self.features = self.cm.features
        self.classes = self.cm.get_classes()
        self.class_distribution = self.cm.class_distribution
        
        self.bid = 0.42
        self.prices = self.cm.prices

        self.p = self.cm.aggr_conv_rates()

        self.n_arms = len(self.prices)
        self.opt = np.max(np.multiply(self.p, self.prices)) 

        self.T = 360 # number of days
        self.n_experiments = 1

        self.reward_log = []
        self.reward_per_experiments = []
        self.regret_per_experiments = []

        self.colors = self.cm.colors
        self.splits = None

        np.random.seed(123)

    def run(self):
        pg = PersonGenerator()

        for e in tqdm(range(0, self.n_experiments)):
            env = SpecificEnvironment(n_arms=self.n_arms, candidates=self.prices)
            context_gen = ContextGenerator([], self.n_arms, self.classes, self.features, self.prices)
            rewards = np.array([])
            regrets = np.array([])

            for t in range(0,self.T): # 1 round is one day
                logging.debug(f'Experiment4.run() -> step {t} / {self.T}')

                if t%14 == 0 and t > 0:
                    context_gen.init_context()
                    context_gen.generate() 

                num_people = pg.generate_people_num(n=500)
                #people = pg.generate_people()
                daily_reward = 0
                daily_regret = 0

                for _ in range(num_people): # p is a class e.g. ["Y", "I"], usually called user_class
                    
                    p_class, p_labels = pg.generate_person()
                
                    pulled_arm, _ = context_gen.pull_arm(p_labels)
                    reward = env.round(pulled_arm, p_class)

                    current_opt = np.max(np.multiply(self.cm.conv_rates[p_class], self.prices))

                    new_obs = [p_labels, pulled_arm, reward]
                    context_gen.update(new_obs)
                    
                    daily_reward += reward
                    daily_regret += current_opt - reward

                rewards = np.append(rewards, daily_reward)
                regrets = np.append(regrets, daily_regret)
  
            self.reward_per_experiments.append(rewards)
            self.regret_per_experiments.append(regrets)

            self.splits = context_gen.get_context_color_matrices()

    def _plot_splits(self):
        if len(self.splits) > 0:
            if self.splits is not None:
                fig, axes = plt.subplots(figsize=(8,4), ncols=len(self.splits))
                for i, s in enumerate(self.splits):
                    split_matrix = np.ndarray(shape=(2,2))
                    for idx1, f1 in enumerate(self.features[0]):
                        for idx2, f2 in enumerate(self.features[1]):
                            split_matrix[idx1, idx2] = s[f1+f2]                
                    axes[i].imshow(split_matrix, alpha=0.8, cmap='magma')

                    axes[i].set_xticks([0, 1])
                    axes[i].set_xticklabels(self.features[0])
                    axes[i].set_yticks([0, 1])
                    axes[i].set_yticklabels(self.features[1])
                    axes[i].set_xlabel('Feature 1')
                    if i == 0:
                        axes[i].set_ylabel('Feature 2')
                    axes[i].set_title(f"{s['obs']} obs")
                fig.suptitle('Context splits after')
                plt.savefig(f'img/experiments/experiment_4_splits.png')

    
    def plot(self):
        self._plot_splits()

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.regret_per_experiments, axis=0)), label='Context Gen', color=self.colors[1])
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_4.png")