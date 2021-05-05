import numpy as np


class PricingEnvironment:
    '''Pricing Environment class.'''
    
    def __init__(self, n_arms, prices, p):
        '''Initialize the Pricing Environment class with a number of arms, a list of prices, 
        a list of conversion rate curves for each subcampaign and the current subcampaign.'''
        
        # Assignments and Initializations
        self.n_arms = n_arms
        self.probabilities = self.conv_rate(prices)
        self.prices = prices
        
    def round(self, pulled_arm):
        '''Simulates the reward as a Bernoulli considering the current probabilities for the pulled arm.'''
        
        # print(self.probabilities)
        # print(self.prices)
        # print(self.prices[pulled_arm])
        # print(self.probabilities(self.prices[pulled_arm]), pulled_arm)
        
        # The reward is Bernoulli with probability based on the conversion rate curve for the current pulled arm
        reward = np.random.binomial(1, self.probabilities(self.prices[pulled_arm]))
        return reward

    def conv_rate(x, a=1, b=1, c=1):
        return ((c*x) ** a) * np.exp(-b * c * x)

    def cost_per_click(bid, alpha):
        beta = np.sqrt(bid)
        return bid * np.random.beta(alpha, beta, 1)


    def return_probability(_lambda, size=100000):
        return np.random.poisson(_lambda, size=size)

    def new_clicks(bid, Na=10000, p0=0.01, cc=0.44):
        p = 1-(cc/(2*bid))
        mean = Na*p*p0
        sd = (Na*p*p0*(1-p0))**0.5
        return np.random.normal(mean,sd)