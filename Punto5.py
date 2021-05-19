from learners import GPTS_learner_positive
from pricing import BiddingEvironment
import numpy as np
import matplotlib.pyplot as plt

n_arms = 20
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 40
n_experiments = 10

gts_reward_per_experiment = []
gpts_reward_per_experiment = []

for e in range(n_experiments):
  env = BiddingEvironment(bids, sigma)
  gpts_learner = GPTS_learner_positive(n_arms=n_arms, arms=bids, threshold=0.2) # qui metto anche bid perchè per implementare GP serve sapere le distanze tra i dati

  for t in range(T):

    pulled_arm = gpts_learner.pull_arm()
    reward = env.round(pulled_arm)
    gpts_learner.update(pulled_arm, reward)
    print(pulled_arm)

  gpts_reward_per_experiment.append(gpts_learner.collected_rewards)

opt = np.max(env.means)
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.cumsum(np.mean(opt - gpts_reward_per_experiment, axis = 0)),'g')

plt.show()