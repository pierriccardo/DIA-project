import logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm, beta


class Learner:

    def __init__(self, n_arms):
        self.n_arms = n_arms

        # current round value
        self.t = 0

        # list of lists to store the collected
        # rewards for each round and for each arm
        # length of the external list: n_arms
        # length of internal list: num of times
        # we pulled a given arm
        self.rewards_per_arm = [[] for i in range(n_arms)]

        # value of the rewards for each round
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        update the collected_rewards and 
        rewards_per_arm

        param pulled_arm: arm pulled in this round
        param reward: reward given by environment
        """

        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        if (self.t < self.n_arms):
            return self.t
        idxs = np.argwhere(self.expected_rewards ==
                           self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (
            self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t


class UCB1(Learner):
    def __init__(self, n_arms, prices, alpha):
        super().__init__(n_arms)
        self.alpha = alpha
        self.empirical_means = np.zeros(n_arms)
        self.buyer = np.zeros(n_arms)
        self.not_buyer = np.zeros(n_arms)
        self.times_pulled = np.zeros(n_arms)
        self.prices = prices
        self.t = 0

    def pull_arm(self):
        # All'inizio devo provare una volta tutti gli arm
        if self.t < self.n_arms:
            arm = self.t
        else:
            
            confidence = np.sqrt(2*np.log(self.t)/(self.buyer+self.not_buyer))
            
            upper_bound = (self.empirical_means + confidence)*self.prices
            arm = np.random.choice(
                np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm] = np.append(
            self.rewards_per_arm[pulled_arm], reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (
            self.empirical_means[pulled_arm]*(self.t) + reward) / self.t

        # I need to update for all the arms because i have t at the denominator
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5
        np.append(self.rewards_per_arm[pulled_arm], reward)

    def update_more(self, pulled_arm, buyer, not_buyer):
        self.t += 1

        self.buyer[pulled_arm] += buyer
        self.not_buyer[pulled_arm] += not_buyer
        self.empirical_means = self.buyer/(self.buyer + self.not_buyer)

        self.times_pulled[pulled_arm] += 1


class TS_Learner(Learner):

    def __init__(self, n_arms, candidates):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.candidates = candidates

    def pull_arm(self):

        beta_samples = np.random.beta(
            self.beta_parameters[:, 0], self.beta_parameters[:, 1])
        expected_rewards = np.multiply(beta_samples, self.candidates)

        idx = np.argmax(expected_rewards)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1

        binary_reward = 1.0 if reward > 0 else 0.0

        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm,
                             0] = self.beta_parameters[pulled_arm, 0] + binary_reward
        self.beta_parameters[pulled_arm,
                             1] = self.beta_parameters[pulled_arm, 1] + 1.0 - binary_reward

    def update_more(self, pulled_arm, buyer, not_buyer):
        self.t += 1
        self.beta_parameters[pulled_arm,
                             0] = self.beta_parameters[pulled_arm, 0] + buyer
        self.beta_parameters[pulled_arm,
                             1] = self.beta_parameters[pulled_arm, 1] + not_buyer

    def success_prob(self, arm):
        # alpha: successes of the arm
        # beta: failures of the arm
        a_arm = self.beta_parameters[arm, 0]
        b_arm = self.beta_parameters[arm, 1]

        # alpha + beta = total attempts
        # success probability is given by
        # successes / total attempts
        return a_arm / (a_arm + b_arm)

    def optimal_arm(self):
        # returns the optimal arm

        # values array store the expected value of each arm
        values = []
        for arm in range(self.n_arms):
            values.append(self.expected_value(arm))

        #print(f'opt value: {np.max(values)}')
        #print(f'opt arm  : {opt_arm}')

        # values array has length n_arms
        # taking the array index of the highest expected
        # value we return the optimal arm
        return values.index(max(values))

    def expected_value(self, arm):
        # returns the expected value of the arm
        # that is its success probability multiplied
        # by its candidate value (price in our case)

        return self.success_prob(arm) * self.candidates[arm]

    def expected_value_lower_bound(self):

        opt_arm = self.optimal_arm()
        exp_val = self.success_prob(opt_arm)
        # Hoeffding bound

        confidence = np.log(0.05)

        alpha = self.beta_parameters[opt_arm, 0]
        beta = self.beta_parameters[opt_arm, 1]

        lb = exp_val - np.sqrt(- confidence / (2 * (alpha + beta)))
        logging.debug(
            f'TS_learner.expected_value_lowerbound() -> lb: {lb}, self.t {self.t}')

        return lb 


class GPTS(Learner):

    def __init__(self, n_arms, arms, threshold):

        super().__init__(n_arms)
        self.arms = arms
        
        self.means = np.zeros(n_arms) 
        self.sigmas = np.ones(n_arms) * 10

        self.pulled_arms = []         # per avere il numero del round utilizzeremo len(pulled_arm)
        self.threshold = threshold
        
        alpha = 3.0

        # Gaussian Process Regressor
        theta = 0.0001
        l = 0.01
        kernel = C(theta, (1e-6, 2e2)) * RBF(l, (1e-6, 1e1))
        self.gp = GaussianProcessRegressor(
            kernel = kernel, 
            alpha = alpha**2, 
            n_restarts_optimizer = 9,
            normalize_y=True)

        self.exp_cost = np.zeros(n_arms)
        self.upper_bound_cost = np.zeros(n_arms)


    def update_observation(self, pulled_arm, reward):
        self.update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observation(pulled_arm, reward)
        self.update_model()

    def is_eligible(self, idx, price_value, exp_cost):
        
        if (price_value-exp_cost < 0):
            return False
        return True

    def pull_arm(self, price_value):
        if (len(self.pulled_arms) < 10):
            return len(self.pulled_arms) #np.random.choice(self.n_arms)   # scelta uniforme nei primi 20 round  --> deve essere coerente con l'enviroment
        sample = np.random.normal(self.means,self.sigmas)
        sample = sample*(price_value - self.exp_cost) # adjust sample wrt price value
        for i in range(len(sample)):  # controllo uno alla volta gli elementi del sample
            idx = np.argmax(sample)
            if self.is_eligible(idx, price_value, self.upper_bound_cost[idx]):
                return idx
            else:
                sample[idx] = -10000.0    # siamo sicuri che nella prossima iterazione non si sceglie?? questo braccio 
        print('errore, nessun braccio eligible, ne restituisco uno a caso')   
        return np.argmax(np.random.normal(self.means,self.sigmas))
