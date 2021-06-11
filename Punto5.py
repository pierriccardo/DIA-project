from learners import GPTS_learner_positive
from pricing import BiddingEvironment, conv_rate
import numpy as np
import matplotlib.pyplot as plt
import yaml
#import seaborn as sns
from tqdm import tqdm
from configmanager import *

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

cm = ConfigManager()

n_arms = 10
bids = config["bids"]
bids = np.array(bids)
#bids = [.30, .32, .34, .36, .38, .40, .42, .44, .46, .48] 
#min_bid = 0.0
#max_bid = 1.0
#bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10
prices = config["prices"]
prices = np.array(prices)
p = cm.aggr_conv_rates()
opt_pricing = np.max(np.multiply(p, prices)) 
print(opt_pricing)

T = 360
n_experiments = 10

gpts_reward_per_experiment = []
p_arms = []

for e in tqdm(range(n_experiments)):
  env = BiddingEvironment(bids, sigma, opt_pricing)
  gpts_learner = GPTS_learner_positive(n_arms=n_arms, arms=bids, threshold=0.2) # qui metto anche bid perchè per implementare GP serve sapere le distanze tra i dati

  for t in range(T):

    pulled_arm = gpts_learner.pull_arm()
    reward = env.round(pulled_arm)
    gpts_learner.update(pulled_arm, reward)
    p_arms.append(pulled_arm)
    #print(pulled_arm)

  gpts_reward_per_experiment.append(gpts_learner.collected_rewards)
  print(gpts_learner.means)

print(p_arms)
#sns.distplot(np.array(p_arms))

plt.hist(p_arms)

opt = np.max(env.means * (opt_pricing - bids))
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.cumsum(np.mean(opt - gpts_reward_per_experiment, axis = 0)),'g')

plt.show()