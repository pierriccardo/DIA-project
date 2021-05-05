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

    def conv_rate(x, a=1.0, b=1.0, c=1.0):
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
        X = 2.21
        N = 10000*0.01*(1-(0.44/(2*0.5)))
        c = self.bid[5] * np.random.beta(self.alpha, np.sqrt(self.bid[5]), 1)
        p = self.prices[pulled_arm]
        reward = (X + 1)*self.conv_rate(float(p))* p * N  - N * c
        return reward


    