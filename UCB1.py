# Here we implement the UCB1 algorithm

#from Learner import Learner 
import numpy as np



"""
a learner object is defined by:

- number of arms that he/she can pull
- the current round
- the list of collected rewards

the learner interacts with the environment by selecting the arm
to pull at each round and observing the reward given by the 
environment
"""
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





class UCB1(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)

    def pull_arm(self):
        # All'inizio devo provare una volta tutti gli arm 
        if self.t < self.n_arms:
            arm = self.t
        else:
            upper_bound = self.empirical_means + self.confidence
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm

    def update_observations(self, pulled_arm, reward):
        self.t += 1
        self.rewards_per_arm[pulled_arm] = np.append(self.rewards_per_arm[pulled_arm], reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t) + reward) / self.t

        # I need to update for all the arms because i have t at the denominator
        for a in range(self.n_arms):
            number_pulled = max(1, len(self.reward_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t)/ number_pulled)**0.5
        self.reward_per_arm[pulled_arm].append(reward)


if __name__ == '__main__':
    from Enviroment_V1 import Environment
    import yaml

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    prices = config["prices"]
    n_arms = len(prices)
    bid = config["bids"]
    alpha = 4.4 
    _lambda = 2.21

    env = Environment(n_arms, prices, bid, alpha, _lambda)
    T = 100
    learner = UCB1(n_arms)
    for _ in range(T):
        pulled_arm = learner.pull_arm()
        print("pulled_arm =", pulled_arm)
        reward = env.round(pulled_arm)
        learner.update_observations(pulled_arm, reward) 
