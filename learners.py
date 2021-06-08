import logging
import numpy as np

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
        # All'inizio devo provare una volta tutti gli arm 
        if self.t < self.n_arms:
            arm = self.t
        else:
            upper_bound = (self.empirical_means + self.confidence)*self.prices
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm] = np.append(self.rewards_per_arm[pulled_arm], reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t) + reward) / self.t

        # I need to update for all the arms because i have t at the denominator
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t)/ number_pulled)**0.5
        np.append(self.rewards_per_arm[pulled_arm], reward)

class TS_Learner(Learner):

    def __init__(self, n_arms, candidates):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.candidates = candidates

    def pull_arm(self):

        beta_samples = np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1])
        expected_rewards = np.multiply(beta_samples, self.candidates)

        idx = np.argmax(expected_rewards)
        return idx

    def update(self, pulled_arm, reward):
        self.t+=1

        binary_reward = 1.0 if reward > 0 else 0.0

        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + binary_reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - binary_reward

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
    
    def expected_value_lower_bound(self, n_obs):

        opt_arm = self.optimal_arm()
        exp_val = self.expected_value(opt_arm)


        succ_prob_opt_arm = self.success_prob(opt_arm)

        confidence = succ_prob_opt_arm / (1 - succ_prob_opt_arm)

        msg = f'|exp_val:{exp_val}|confidence:{confidence}|:{n_obs}|'
        logging.info(f'TS_Learner.expected_value_lower_bound() -> {msg}')       

        # Hoeffding bound
        # TODO: NB: change self.t with len(obs) if we want to use
        # more observations per day
        

        # TODO: fix the lower bound, it doesn't work
        #lb = exp_val - np.sqrt(-np.log(confidence) / (2 * n_obs + 1))
        #logging.error(f'TS_Learner.expected_value_lower_bound() -> error in lower bound computation: {lb}')
        #lb = lb if lb is not np.nan else 0 
        lb = exp_val

        return lb 



