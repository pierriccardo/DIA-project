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
        
        self.classes = self.get_classes()
        self.num_classes = len(self.classes)
        
        self.avg_cc = self.config["avg_cc"] # avg cost per click

        self.features = self.config["features"] # ["Y", "A"], ["I", "D"]

        self.conv_rates = self.config['conv_rates']
        self.colors = self.config['colors']
    
    #------------------------------
    # ENVIRONMENT FUNCTIONS
    #------------------------------

    def return_probability(_lambda, size=1):
        samples = np.random.poisson(_lambda, size=size)

        # return more samples is just for plotting
        return samples if size > 1 else samples[0] 

    def cost_per_click(self, bid, alpha):
        beta = np.sqrt(bid)
        return bid * np.random.beta(alpha, beta, 1)
    
    def new_clicks(self, bids):
        return 100*(1.0-np.exp(-4*bids+3*bids**3))
    
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
            aggr_cr = np.add(aggr_cr, self.conv_rates[c])
        
        return np.divide(aggr_cr, len(classes))
    
    def aggr_return_probability(self):
        ret_prob = 0
        for user_class in self.classes:
            _lambda = self.config["return_probability"][user_class]

            ret_prob += self.return_probability(_lambda)
        return ret_prob / self.num_classes
    
    def aggr_cost_per_click(self, bid):
        aggr_cc = 0
        for user_class in self.classes:
            alpha = self.config["cost_per_click"][user_class]
            aggr_cc += self.cost_per_click(bid, alpha)
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

def cost_per_click(bid, alpha):
    beta = np.sqrt(bid)
    return bid * np.random.beta(alpha, beta, 1)

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


