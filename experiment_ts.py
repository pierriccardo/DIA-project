import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from learners import TS_Learner, Greedy_Learner, UCB1, TS_Learner_Prices
from tqdm import tqdm
import yaml
from pricing import *

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

bid = 0.42
prices = config["prices"]

avg_cc = config["avg_cc"]

# preparing the environment
aggregated = config["aggregated"]

# conv rate
a, b, c = tuple(aggregated["conv_rate"])
p = [conv_rate(p, a, b, c) for p in prices]

# return probability
returns = return_probability(aggregated["return_probability"])

# new clicks
Na, p0 = tuple(aggregated["new_clicks"])
Nc = new_clicks(bid, Na, p0, avg_cc)

# cost per click
cc = cost_per_click(bid, aggregated["cost_per_click"])

n_arms = len(config["prices"])
opt = p[4]


T = 300 # number of days


# number of exp to perform
n_experiments = 100
ts_reward_per_experiments = []
gr_reward_per_experiments = []
uc_reward_per_experiments = []
tp_reward_per_experiments = []

for e in tqdm(range(0, n_experiments)):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms)
    gr_learner = Greedy_Learner(n_arms=n_arms)
    uc_learner = UCB1(n_arms=n_arms, prices=config['prices'])
    tp_learner = TS_Learner_Prices(n_arms=n_arms)
    for t in range(0,T):
        #Thompson sampling learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        
        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

        # UCB1
        pulled_arm = uc_learner.pull_arm()
        reward = env.round(pulled_arm)
        uc_learner.update(pulled_arm, reward)

        # TS prices
        pulled_arm = tp_learner.pull_arm(config['prices'])
        reward = env.round(pulled_arm)
        tp_learner.update(pulled_arm, reward)

    ts_reward_per_experiments.append(ts_learner.collected_rewards)
    gr_reward_per_experiments.append(gr_learner.collected_rewards)
    uc_reward_per_experiments.append(uc_learner.collected_rewards)
    tp_reward_per_experiments.append(tp_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_reward_per_experiments, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_reward_per_experiments, axis=0)), 'g')
plt.plot(np.cumsum(np.mean(opt - uc_reward_per_experiments, axis=0)), 'y')
plt.plot(np.cumsum(np.mean(opt - tp_reward_per_experiments, axis=0)), 'b')
plt.legend(["TS", "Greedy", "UCB1", "TS_prices"])
plt.savefig("exp.png")
#plt.show()