import logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
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
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t+=1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t

class UCB1(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)
        self.prices = prices

    def pull_arm(self):
        # at start we try all the arms once
        if self.t < self.n_arms:
            arm = self.t
        else:
            upper_bound = (self.empirical_means + self.confidence)*self.prices
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        np.append(self.rewards_per_arm[pulled_arm], reward)
        np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t) + reward) / self.t

        # I need to update for all the arms because i have t at the denominator
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = np.sqrt(2*np.log(self.t) / number_pulled)

class TS_Learner(Learner):

    def __init__(self, n_arms, candidates):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.candidates = candidates

    def pull_arm(self):
        
        beta_samples = np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])
        expected_rewards = np.multiply(beta_samples, self.candidates)

        idx = np.argmax(expected_rewards)
        return idx, max(expected_rewards)

    def update(self, pulled_arm, reward):
        self.t+=1

        binary_reward = 1.0 if reward > 0 else 0.0

        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + binary_reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - binary_reward
    
    def update_more(self, pulled_arm, reward, buyer, not_buyer):
        self.t+=1

        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + buyer
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + not_buyer

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

        #logging.debug(f'TS_Learner.expected_value_lowerbound() -> opt arm: {opt_arm}, exp_val: {exp_val}')

        # Hoeffding bound
        # TODO: NB: change self.t with len(obs) if we want to use
        # TODO: chenage n_obs with n_obs of the optimal arm only
        # more observations per day
        
        confidence = np.log(0.005)

        # TODO: fix the lower bound, it doesn't work
        #lb = exp_val - np.sqrt(-np.log(alpha) / (self.t + 1))
        #logging.error(f'TS_Learner.expected_value_lower_bound() -> error in lower bound computation: {lb}')
        #confidence = self.success_prob(opt_arm) / (1 + self.success_prob(opt_arm)) 
        lb = exp_val - np.sqrt(- confidence / (2*self.t) + 0.001)
        logging.debug(f'TS_learner.expected_value_lowerbound() -> lb: {lb}, self.t {self.t}')

        return lb * self.candidates[opt_arm]
    
class GPTS_learner_positive(Learner):

    def __init__(self, n_arms, arms, threshold):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.ones(n_arms)*10000 #np.zeros(n_arms)
        self.sigmas = np.ones(n_arms)*10000
        self.pulled_arms = []         # per avere il numero del round utilizzeremo len(pulled_arm)
        self.threshold = threshold
        alpha = 3.0 #10.0
        kernel = C(0.0001, (1e-6, 2e1)) * RBF(0.01, (1e-6, 1e1  ))
        self.gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, n_restarts_optimizer = 9)


    def UpdateObservation(self, idx, reward):
        self.update_observations(idx, reward)
        self.pulled_arms.append(self.arms[idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std = True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.UpdateObservation(pulled_arm, reward)
        self.update_model()

    def is_eligible(self, idx, price_value):
        proba = norm(loc = self.means[idx], scale = self.sigmas[idx]).cdf(0.0)
        if (proba > self.threshold):
            return False
        if (price_value-self.arms[idx]*beta.ppf(1-self.threshold, 4.4, self.arms[idx]**0.5) < 0):
            return False
        return True

    def pull_arm(self, price_value):
        if (len(self.pulled_arms) < 10):
            np.random.seed(123)
            return np.random.choice(self.n_arms)   # scelta uniforme nei primi 20 round  --> deve essere coerente con l'enviroment
        np.random.seed(123)
        sample = np.random.normal(self.means,self.sigmas)
        sample = sample*(price_value - self.arms*4.44/(4.4 + self.arms**0.5)) # adjust sample wrt price value
        neg = []
        for i in range(len(sample)):  # controllo uno alla volta gli elementi del sample
            idx = np.argmax(sample)
            if self.is_eligible(idx, price_value):
                return idx
            else:
                print(idx)
                sample[idx] = -10000.0    # siamo sicuri che nella prossima iterazione non si sceglieà questo braccio 
            print('errore, nessun braccio eligible, ne restituisco uno a caso')    
        return np.argmax(np.random.normal(self.means,self.sigmas))