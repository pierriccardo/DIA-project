import numpy as np
import yaml
import itertools
from utils import *
import logging

class ConfigManager():

    def __init__(self):

        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.bids = self.config['bids']
        self.prices = self.config["prices"] # candidates
        self.n_arms = len(self.prices)

        self.class_distribution = self.config['class_distribution']
        self.new_clicks = self.config["new_clicks"]
        
        self.classes = self.get_classes()
        self.num_classes = len(self.classes)
        
        self.avg_cc = self.config["avg_cc"] # avg cost per click
        self.cc = self.config['cost_per_click']

        self.ret = self.config['return_probability']

        self.avg_ret = np.mean(self.config['return_probability'])

        self.features = self.config["features"] # ["Y", "A"], ["I", "D"]

        self.conv_rates = self.config['conv_rates']
        self.colors = self.config['colors']
    
    #------------------------------
    # ENVIRONMENT FUNCTIONS
    #------------------------------

    def mean_ret(self, classes):
        aggr_ret = 0
        for c in classes:
            ret_scaled = [self.ret[c][i]*self.class_distribution[c] for i in range(len(self.ret[c]))]
            aggr_ret = np.add(aggr_ret, ret_scaled)
            #aggr_ret += self.config['return_probability'][c]
        return aggr_ret #/ len(classes)

    def return_probability(_lambda, size=1):
        samples = np.random.poisson(_lambda, size=size)

        # return more samples is just for plotting
        return samples if size > 1 else samples[0] 

    def cost_per_click(self, bid, classe, size, mean = False):
        beta = np.sqrt(bid)
        alpha = self.config['cost_per_click'][classe]
        if (mean):
            return bid*alpha/(alpha+beta)
        return bid * np.random.beta(alpha, beta, size)
    
    #def new_clicks(self, bids):
    #    return 100*(1.0-np.exp(-4*bids+3*bids**3))

    def new_clicks_function_mean(self, bid, classe, num_people):
        return (1-0.40/(2*bid))*num_people*self.new_clicks[classe]
    
    def aggregated_new_clicks_function_mean(self, bid, num_people):
        v = 0
        for i in range(4):
            v += self.new_clicks_function_mean(bid, i, num_people[i])
        return v

    def new_clicks_function_sigma(self, bid, classe, num_people):
        return (1-0.40/(2*bid))*num_people**0.5*self.new_clicks[classe]

    def aggregated_new_clicks_function_sigma(self, bid, num_people):
        v = 0
        for i in range(4):
            v += self.new_clicks_function_sigma(bid, i, num_people[i])
        return v

    def cc(self, bid, size = 1):
        return bid*np.random.beta(4.4,bid**0.5, size=size)
    
    def mean_cc(self, bid):
        return bid*(4.4/(4.4+bid**0.5))
    #------------------------------
    # AGGREGATED FUNCTIONS
    #------------------------------
        
    def aggr_conv_rates(self, classes=[0,1,2,3]):    
        '''
        return the aggregated conversion rate of the specified classes
        classes is a vector of type [0, 1, 2, 3]
        if we want to aggregate just 2 classes [0, 3]
        '''
        aggr_cr = np.zeros(self.n_arms)
        for c in classes:
            conv_rate_scaled = [self.conv_rates[c][i]*self.class_distribution[c] for i in range(len(self.conv_rates[c]))]
            aggr_cr = np.add(aggr_cr, conv_rate_scaled)
        
        return aggr_cr
    
    def aggr_return_probability(self):
        ret_prob = 0
        for user_class in self.classes:
            _lambda = self.config["return_probability"][user_class]

            ret_prob += self.return_probability(_lambda)
        return ret_prob / self.num_classes
    
    def aggr_cost_per_click(self, bid):
        aggr_cc = 0
        for _ in self.classes:
            aggr_cc += self.cost_per_click(bid)
        return aggr_cc / self.num_classes
    
    #------------------------------
    # GENERIC FUNCTIONS
    #------------------------------
    def get_classes(self):

        classes = []
        features = self.config["features"]
        for f in itertools.product(features[0], features[1]):
            classes.append([f[0], f[1]])
        
        return classes
    
    def get_features(self):
        return self.config["features"]

def conv_rate(x, a=1, b=1, c=1):
        return ((c*x) ** a) * np.exp(-b * c * x)

#def cost_per_click(bid, alpha):
#    beta = np.sqrt(bid)
#    return bid * np.random.beta(alpha, beta, 1)

def return_probability(_lambda, size=1):
    samples = np.random.poisson(_lambda, size=size)
    return samples if size > 1 else samples[0]

def new_clicks(bid, Na=10000, p0=0.01, cc=0.44):
    p = 1-(cc/(2*bid))
    mean = Na*p*p0
    sd = (Na*p*p0*(1-p0))**0.5
    return np.random.normal(mean,sd)

def aggregated_new_cliks(bid, Na=10000, cc=0.44):
    return self.config["frequencies"]["class1"]*new_clicks(bid, Na, new_clicks["class1"][1])+self.config["frequencies"]["class2"]*new_clicks(bid, Na, new_clicks["class2"][1])+self.config["frequencies"]["class3"]*new_clicks(bid, Na, new_clicks["class3"][1])+self.config["frequencies"]["class4"]*new_clicks(bid, Na, new_clicks["class4"][1])

