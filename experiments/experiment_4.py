from learners import TS_Learner
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
     
        self.prices = self.cm.prices
        self.n_arms = len(self.prices)

        for i in range(4):
            logging.debug(f'experiment_4().__init__() -> current_opt_{i} {np.max(np.multiply(self.cm.conv_rates[i], self.prices))}')

        self.reward_per_experiments = []
        self.regret_per_experiments = []

        self.comulative_regret_per_experiment = []

        self.colors = self.cm.colors
        self.splits = None

        self.T = 365 # number of days
        self.n_experiments = 20

        self.ts_reward_per_experiments = []
        self.ts_regret_per_experiment = []
        self.ts_comulative_regret_per_experiment = []
        self.opt_per_experiment = []

    def run(self):
        pg = PersonGenerator()
        observed_class = [0, 0, 0, 0]

        for e in tqdm(range(0, self.n_experiments)):
            env = SpecificEnvironment(n_arms=self.n_arms, candidates=self.prices)
            context_gen = ContextGenerator([], self.n_arms, self.classes, self.features, self.prices)

            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)
            rewards = np.array([])
            regrets = np.array([])

            ts_regret = []

            for t in range(0,self.T): # 1 round is one day
                logging.debug(f'Experiment4.run() -> step {t} / {self.T}')

                if t%7 == 0 and t > 0:
                    context_gen.init_context()
                    context_gen.generate() 

                num_people = pg.generate_people_num(n=300)
               
                daily_reward = 0
                daily_regret = 0
                

                # pull arms from context, for each context we pull
                # the arm, then according to the class of the generated
                # person we select the arm of the context containing
                # that class, pulled arms is an array with elements like:
                # [[context classes, pulled_arm] , ...]
                pulled_arms = context_gen.pull_arm()

                # TS
                ts_buyers = 0
                ts_daily_reward = 0
                ts_pulled_arm, _ = ts_learner.pull_arm()
                ts_daily_opt = 0

                for _ in range(num_people):                         
                    # we generate a new customer where
                    # p_class is a number in [0,1,2,3] i.e. the class
                    # p_labels is the correspondent class e.g. ['Y', 'I']
                    p_class, p_labels = pg.generate_person()

                    # this is just for debugging purpose, we store for each
                    # class the number of people that come
                    observed_class[p_class] += 1
                    
                    pulled_arm = None
                    for pulled in pulled_arms:
                        if p_labels in pulled[0]:
                            pulled_arm = pulled[1]

                    reward = env.round(pulled_arm, p_class)
                    

                    # TODO: make it an array without recompute the whole thing
                    current_opt = np.max(np.multiply(self.cm.conv_rates[p_class], self.prices))
                    
                    
                    new_obs = [p_labels, pulled_arm, reward]
                    context_gen.update(new_obs)
                    
                    daily_reward += reward
                    daily_regret += (current_opt - reward)

                    # TS
                    ts_reward = env.round(ts_pulled_arm, p_class)
                    ts_daily_reward += ts_reward
                    if ts_reward > 0: ts_buyers += 1
                    ts_daily_opt += current_opt
                ts_learner.update_more(ts_pulled_arm, ts_reward, ts_buyers, num_people - ts_buyers) 
                ts_regret.append(ts_daily_opt - ts_daily_reward)

                rewards = np.append(rewards, daily_reward)
                regrets = np.append(regrets, daily_regret)
  
            self.reward_per_experiments.append(rewards)
            self.regret_per_experiments.append(regrets)
            self.comulative_regret_per_experiment.append(np.cumsum(regrets))

            # TS
            self.ts_regret_per_experiment.append(ts_regret)
            self.ts_comulative_regret_per_experiment.append(np.cumsum(ts_regret)) 

            
            splits_freq = context_gen.compute_split_frequency()
            logging.info(f'contexts generated in experiment {e}:')
            logging.info(f'{context_gen.splits_frequency}')
            logging.info(f'split freqs exp {e} {splits_freq}')
            #print(splits_freq)

            self.splits = context_gen.get_split_matrices()

            for index, label in zip(range(4), self.classes):
                logging.debug(f'experiment_4().run() -> class {label} : people {observed_class[index]}')

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
                fig.suptitle('Context splits')
                plt.savefig(f'img/experiments/experiment_4_splits.png')

    def plot(self):
        #self._plot_splits()
        self.plot_ts()

        plt.figure(1)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.regret_per_experiments, axis=0)), label='Context Gen', color=self.colors[1])
        plt.plot(1.96*np.std(self.comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.regret_per_experiments, axis=0)),self.colors[1],linestyle='dashed')
        plt.plot(-1.96*np.std(self.comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.regret_per_experiments, axis=0)),self.colors[1],linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_4.png")
    
    def plot_ts(self):

        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.regret_per_experiments, axis=0)), label='Context Gen', color=self.colors[1])
        plt.plot(1.96*np.std(self.comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.regret_per_experiments, axis=0)),self.colors[1],linestyle='dashed')
        plt.plot(-1.96*np.std(self.comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.regret_per_experiments, axis=0)),self.colors[1],linestyle='dashed')
        plt.plot(np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)), self.colors[3], label="TS")
        plt.plot(1.96*np.std(self.ts_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)),self.colors[3],linestyle='dashed')
        plt.plot(-1.96*np.std(self.ts_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)),self.colors[3],linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_4_TS.png")