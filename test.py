import numpy as np
import matplotlib.pyplot as plt
from learners import Learner
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm


def fun(x):
  return 100*(1.0-np.exp(-4*x+3*x**3))

class BiddingEvironment():
  def __init__(self,bids,sigma):
    self.bids = bids
    self.means = fun(bids)
    self.sigmas = sigma*np.ones(len(bids))

  def round(self, pulled_arm):    # pulled arm is the index of one of the bids
    return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])



# GPTS modificato in modo da tener conto delle restrizioni del prof

class GPTS_learner_positive(Learner):
  def __init__(self, n_arms, arms, threshold):
    super().__init__(n_arms)
    self.arms = arms
    self.means = np.zeros(n_arms)
    self.sigmas = np.ones(n_arms)*10
    self.pulled_arms = []         # per avere il numero del round utilizzeremo len(pulled_arm)
    self.threshold = threshold
    alpha = 10.0
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
    self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, n_restarts_optimizer = 9)


  def UpdateObservation(self, idx, reward):
    self.update_observations(idx, reward)
    self.pulled_arms.append(self.arms[idx])

  def update_model(self):
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    #print(x)
    #print(y)
    self.gp.fit(x,y)
    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)
    self.sigmas = np.maximum(self.sigmas, 1e-2)

  def update(self, pulled_arm, reward):
    self.t += 1
    self.UpdateObservation(pulled_arm, reward)
    self.update_model()

  def is_eligible(self, idx):
    proba = norm(loc = self.means[idx], scale = self.sigmas[idx]).cdf(0.0)
    if (proba < self.threshold):
      return True
    return False

    
  def pull_arm(self):
    if (len(self.pulled_arms) < 10):
      return np.random.choice(self.n_arms)   # scelta uniforme nei primi 10 round   
    sample = np.random.normal(self.means,self.sigmas)
    for i in range(len(sample)):  # controllo uno alla volta gli elementi del sample
      idx = np.argmax(sample)
      if self.is_eligible(idx):
        return idx
      else:
        sample[idx] = -10000.0    # siamo sicuri che nella prossima iterazione non si sceglieà questo braccio 
    print('errore, nessun braccio eligible, ne restituisco uno a caso')    
    return np.argmax(np.random.normal(self.means,self.sigmas))

n_arms = 20
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 40
n_experiments = 10

gts_reward_per_experiment = []
gpts_reward_per_experiment = []

for e in tqdm(range(n_experiments)):
  env = BiddingEvironment(bids, sigma)
  gpts_learner = GPTS_learner_positive(n_arms=n_arms, arms=bids, threshold=0.2) # qui metto anche bid perchè per implementare GP serve sapere le distanze tra i dati

  for t in range(T):

    pulled_arm = gpts_learner.pull_arm()
    reward = env.round(pulled_arm)
    gpts_learner.update(pulled_arm, reward)
    #print(pulled_arm)

  gpts_reward_per_experiment.append(gpts_learner.collected_rewards)

opt = np.max(env.means)
plt.figure(0)
plt.ylabel('Regret')
plt.xlabel('t')
plt.plot(np.cumsum(np.mean(opt - gpts_reward_per_experiment, axis = 0)),'g')

plt.show()