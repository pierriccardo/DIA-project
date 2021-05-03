import numpy as np

'''
Prova di enviroment 
'''

class Environment():
    def __init__(self, n_arms, prices, bid, alpha, _lambda):

        self.n_arms = n_arms
        self.prices = prices
        self.bid = bid 
        self.alpha = alpha
        self._lambda = _lambda

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

    def round(self, pulled_arm):
        print(self._lambda)
        X = np.random.poisson(2.21, 1)
        print(self.bid[5])
        N = self.new_clicks(self.bid[5])
        c = self.cost_per_click(self.bid[5],self.alpha) 
        reward = (X + 1)*self.conv_rate(self.prices[pulled_arm])*self.prices[pulled_arm] * N  - N * c
        return reward


    