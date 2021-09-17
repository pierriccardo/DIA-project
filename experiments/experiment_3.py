import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

from pricing import PersonGenerator
from environment import Environment
from learners import TS_Learner, Greedy_Learner, UCB1
from tqdm import tqdm
from configmanager import *


class Experiment3():

    NAME = 'Experiment 3'

    def __init__(self, days=365, n_exp=100):
        cm = ConfigManager()

        self.opt_2 = []
        self.opt_2.append(np.max(np.multiply(cm.conv_rates[0], cm.prices)))  ## per il grafico 2
        self.opt_2.append(np.max(np.multiply(cm.conv_rates[1], cm.prices)))  ## per il grafico 2
        self.opt_2.append(np.max(np.multiply(cm.conv_rates[2], cm.prices)))  ## per il grafico 2
        self.opt_2.append(np.max(np.multiply(cm.conv_rates[3], cm.prices)))  ## per il grafico 2
        

        self.prices = cm.prices # candidates

        self.p = cm.aggr_conv_rates()
        self.n_arms = cm.n_arms
        self.opt = np.max(np.multiply(self.p, self.prices)) 

        self.T = days # number of days
        self.n_experiments = n_exp

        self.colors = cm.colors

        self.gr_reward_per_experiments = []
        self.uc_reward_per_experiments = []
        self.ts_reward_per_experiments = []

        self.gr_regret_per_experiment = []
        self.uc_regret_per_experiment = []
        self.ts_regret_per_experiment = []

        self.gr_regret_per_experiment_2 = []
        self.uc_regret_per_experiment_2 = []
        self.ts_regret_per_experiment_2 = []

        self.gr_comulative_regret_per_experiment = []
        self.uc_comulative_regret_per_experiment = []
        self.ts_comulative_regret_per_experiment = []

        self.gr_reward_per_experiment = []
        self.uc_reward_per_experiment = []
        self.ts_reward_per_experiment = []

        #self.opt_per_experiment = []

        self.opt_per_exp = []

    def run(self):
        
        pg = PersonGenerator()

        for e in tqdm(range(0, self.n_experiments)):
            env = Environment(n_arms=self.n_arms, probabilities=self.p, candidates=self.prices)

            gr_learner = Greedy_Learner(n_arms=self.n_arms)
            uc_learner = UCB1(n_arms=self.n_arms, prices=self.prices, alpha=1)
            ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.prices)

            gr_regret = []
            uc_regret = []
            ts_regret = []

            gr_regret_2 = []
            uc_regret_2 = []
            ts_regret_2 = []

            gr_reward_2 = []
            uc_reward_2 = []
            ts_reward_2 = []

            opt_days = []

            observed_class = [0, 0, 0, 0] ## per il grafico 2

            for t in range(0,self.T):

                gr_daily_reward = 0
                uc_daily_reward = 0
                ts_daily_reward = 0

                daily_regret_gr = 0 ## per il grafico 2
                daily_regret_uc = 0 ## per il grafico 2
                daily_regret_ts = 0 ## per il grafico 2

                people = pg.generate_people_num(100)
                daily_opt = self.opt * people

                gr_pulled_arm = gr_learner.pull_arm()
                uc_pulled_arm = uc_learner.pull_arm()
                ts_pulled_arm = ts_learner.pull_arm()

                uc_buyers = 0
                ts_buyers = 0
                for _ in range(people):

                    p_class, p_labels = pg.generate_person() ## per il grafico 2
                    observed_class[p_class] += 1             ## per il grafico 2
                    current_opt = self.opt_2[p_class]  ## per il grafico 2

                    gr_reward = env.round(gr_pulled_arm)
                    uc_reward = env.round(uc_pulled_arm)
                    ts_reward = env.round(ts_pulled_arm)   

                    gr_daily_reward += gr_reward
                    uc_daily_reward += uc_reward
                    ts_daily_reward += ts_reward

                    daily_regret_gr += (current_opt - gr_reward) ## per il grafico 2
                    daily_regret_uc += (current_opt - uc_reward) ## per il grafico 2
                    daily_regret_ts += (current_opt - ts_reward) ## per il grafico 2

                    if uc_reward > 0: uc_buyers += 1
                    if ts_reward > 0: ts_buyers += 1
                
                gr_learner.update(gr_pulled_arm, gr_reward)
                uc_learner.update_more(uc_pulled_arm, uc_buyers, people - uc_buyers)
                ts_learner.update_more(ts_pulled_arm, ts_buyers, people - ts_buyers) 

                gr_regret.append(daily_opt - gr_daily_reward)
                uc_regret.append(daily_opt - uc_daily_reward)
                ts_regret.append(daily_opt - ts_daily_reward)

                gr_regret_2.append(daily_regret_gr) ## per il grafico 2 
                uc_regret_2.append(daily_regret_uc) ## per il grafico 2
                ts_regret_2.append(daily_regret_ts) ## per il grafico 2

                gr_reward_2.append(gr_daily_reward) ## per il grafico 3
                uc_reward_2.append(uc_daily_reward) ## per il grafico 3
                ts_reward_2.append(ts_daily_reward) ## per il grafico 3

                opt_days.append(daily_opt)


            self.gr_regret_per_experiment.append(gr_regret)
            self.uc_regret_per_experiment.append(uc_regret)
            self.ts_regret_per_experiment.append(ts_regret)

            self.gr_regret_per_experiment_2.append(gr_regret_2) ## per il grafico 2
            self.uc_regret_per_experiment_2.append(uc_regret_2) ## per il grafico 2
            self.ts_regret_per_experiment_2.append(ts_regret_2) ## per il grafico 2

            self.gr_reward_per_experiment.append(gr_reward_2)
            self.uc_reward_per_experiment.append(uc_reward_2)
            self.ts_reward_per_experiment.append(ts_reward_2)

            self.gr_comulative_regret_per_experiment.append(np.cumsum(gr_regret))
            self.uc_comulative_regret_per_experiment.append(np.cumsum(uc_regret))
            self.ts_comulative_regret_per_experiment.append(np.cumsum(ts_regret)) 
            #print(self.gr_comulative_regret_per_experiment)  
            # 
            self.opt_per_exp.append(opt_days) 

            
            


    def plot(self):

        plt.figure(30)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.gr_regret_per_experiment, axis=0)), self.colors[0], label="Greedy")
        plt.plot(1.96*np.std(self.gr_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.gr_regret_per_experiment, axis=0)), self.colors[0],linestyle='dashed',label="Greedy Confidence Interval 95%")
        plt.plot(-1.96*np.std(self.gr_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.gr_regret_per_experiment, axis=0)), self.colors[0],linestyle='dashed')
        plt.plot(np.cumsum(np.mean(self.uc_regret_per_experiment, axis=0)), self.colors[1], label="UCB1")
        plt.plot(1.96*np.std(self.uc_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.uc_regret_per_experiment, axis=0)),self.colors[1],linestyle='dashed',label="UCB1 Confidence Interval 95%")
        plt.plot(-1.96*np.std(self.uc_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.uc_regret_per_experiment, axis=0)),self.colors[1],linestyle='dashed')
        plt.plot(np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)), self.colors[3], label="TS")
        plt.plot(1.96*np.std(self.ts_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)),self.colors[3],linestyle='dashed',label="TS Confidence Interval 95%")
        plt.plot(-1.96*np.std(self.ts_comulative_regret_per_experiment, axis=0)/np.sqrt(self.n_experiments) + np.cumsum(np.mean(self.ts_regret_per_experiment, axis=0)),self.colors[3],linestyle='dashed')
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_3.png")

        plt.figure(31)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(np.mean(self.gr_regret_per_experiment_2, axis=0)), self.colors[0], label="Greedy")
        plt.plot(np.cumsum(np.mean(self.uc_regret_per_experiment_2, axis=0)), self.colors[1], label="UCB1")
        plt.plot(np.cumsum(np.mean(self.ts_regret_per_experiment_2, axis=0)), self.colors[3], label="TS")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_3_1.png")

        plt.figure(32)
        plt.xlabel("t")
        plt.ylabel("Reward")
        plt.plot(np.mean(self.gr_reward_per_experiment, axis=0), self.colors[0], label="Greedy")
        plt.plot(np.mean(self.uc_reward_per_experiment, axis=0), self.colors[1], label="UCB1")
        plt.plot(np.mean(self.ts_reward_per_experiment, axis=0), self.colors[3], label="TS")
        plt.plot(np.mean(self.opt_per_exp, axis=0), self.colors[2], label="Opt")
        plt.legend(loc=0)
        plt.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        plt.savefig("img/experiments/experiment_3_2.png")