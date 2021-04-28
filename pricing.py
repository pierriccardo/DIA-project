import numpy as np


def conv_rate(x, a=1, b=1):
    return (x ** a) * np.exp(-b * x)

def cost_per_click(bid, alpha):
    beta = np.sqrt(bid)
    return bid * np.random.beta(alpha, beta, 1)