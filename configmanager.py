import numpy as np
import yaml
import itertools
from utils import *
import logging

class ConfigManager():

    def __init__(self):

        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.classes = self.config["classes"]
        self.num_classes = len(self.classes)
        self.prices = self.config["prices"] # price candidates
        self.avg_cc = self.config["avg_cc"] # avg cost per click

        self.features = self.config["features"]
        self.class_distribution = self.config["class_distribution"]
    
    #------------------------------
    # ENVIRONMENT FUNCTIONS
    #------------------------------

    def conv_rate(self, x, a=1, b=1, c=1):
        return ((c*x) ** a) * np.exp(-b * c * x)

    def return_probability(_lambda, size=1):
        samples = np.random.poisson(_lambda, size=size)

        # return more samples is just for plotting
        return samples if size > 1 else samples[0] 

    def cost_per_click(self, bid, alpha):
        beta = np.sqrt(bid)
        return bid * np.random.beta(alpha, beta, 1)
    
    #------------------------------
    # AGGREGATED FUNCTIONS
    #------------------------------

    def class_conv_rate(self, c):
        # c is the class
        a,b,c = tuple(self.config["conv_rate"][c[0]+c[1]])          
        conv_rates = [self.conv_rate(p, a, b, c) for p in self.prices]
        
    def aggr_conv_rates(self, num_candidates=10):    

        aggr_conv_rate = np.zeros(num_candidates)

        for user_class in self.classes:
            a,b,c = tuple(self.config["conv_rate"][user_class])          
            conv_rates = [self.conv_rate(p, a, b, c) for p in self.prices]
            aggr_conv_rate = np.add(aggr_conv_rate, conv_rates)
        
        return np.divide(aggr_conv_rate, self.num_classes)
    
    def class_aggr_conv_rates(self, classes):
        aggr_conv_rate = np.zeros(len(self.prices))

        for c in classes:
            a,b,c = tuple(self.config["conv_rate"][c[0]+c[1]])          
            conv_rates = [self.conv_rate(p, a, b, c) for p in self.prices]
            aggr_conv_rate = np.add(aggr_conv_rate, conv_rates)
        
        return np.divide(aggr_conv_rate, len(classes))

    
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


