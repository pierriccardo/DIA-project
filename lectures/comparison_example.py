import numpy as np
import matplotlib.pyplot as plt
from Environment import *
from learners.TS_Learner import *
from learners.Greedy_Learner import *
from tqdm import tqdm

n_arms = 4
p = np.array([ 0.15, 0.1, 0.1, 0.35])
opt = p[3]

T = 300

# number of exp to perform
n_experiments = 1000
ts_reward_per_experiments = []
gr_reward_per_experiments = []


for e in tqdm(range(0, n_experiments)):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_Learner(n_arms=n_arms)
    gr_learner = Greedy_Learner(n_arms=n_arms)
    for t in range(0,T):
        #Thompson sampling learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)
    
    ts_reward_per_experiments.append(ts_learner.collected_rewards)
    gr_reward_per_experiments.append(gr_learner.collected_rewards)


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_reward_per_experiments, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_reward_per_experiments, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.savefig("exp.png")
plt.show()
