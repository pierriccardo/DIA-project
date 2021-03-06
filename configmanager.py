import numpy as np
import yaml
import itertools
import logging

class ConfigManager():

    def __init__(self):

        with open('config.yml', 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.exp_values = self.config['exp_values']

        self.num_people = self.config['num_people']
        self.env_img_path = self.config["env_imgpath"]
        self.class_labels = self.config["class_labels"] 
        
        self.bids = self.config['bids']
        self.prices = self.config["prices"] # candidates
        self.n_arms = len(self.prices)

        self.class_distribution = self.config['class_distribution']
        self.new_clicks_lambdas = self.config["new_clicks"]
        
        self.classes = self.get_classes()
        self.num_classes = len(self.classes)
        
        self.avg_cc = self.config["avg_cc"] # avg cost per click
        self.costo = self.config['cost_per_click']

        self.ret = self.config['return_probability']

        self.avg_ret = np.mean(self.config['return_probability'])

        self.features = self.config["features"] # ["Y", "A"], ["I", "D"]

        self.conv_rates = self.config['conv_rates']
        self.colors = self.config['colors']
    
    #------------------------------
    # ENVIRONMENT FUNCTIONS
    #------------------------------

    def return_probability(self, lam, size=1):
        samples = np.random.poisson(lam, size=size)

        # return more samples is just for plotting
        return samples if size > 1 else samples[0] 

    def cost_per_click(self, bid, classe, size, mean = False):
        beta = np.sqrt(bid)
        alpha = self.config['cost_per_click'][classe]
        if (mean):
            return bid*alpha/(alpha+beta)
        return bid * np.random.beta(alpha, beta, size)

    def new_clicks(self, bid, user_class):
        num_people = self.num_people * self.class_distribution[user_class]
        mean = self.new_clicks_function_mean(bid, user_class, num_people)
        sigma = self.new_clicks_function_sigma(bid, user_class, num_people)
        return np.random.normal(mean,sigma)
       
    def new_clicks_function_mean(self, bid, classe, num_people):
        return (1-0.40/(2*bid))*num_people*self.new_clicks_lambdas[classe]
    
    def aggregated_new_clicks_function_mean(self, bid, num_people):
        v = 0
        for i in range(4):
            v += self.new_clicks_function_mean(bid, i, num_people[i])
        return v

    def new_clicks_function_sigma(self, bid, classe, num_people):
        return (1-0.40/(2*bid))*num_people**0.5*self.new_clicks_lambdas[classe]

    def aggregated_new_clicks_function_sigma(self, bid, num_people):
        v = 0
        for i in range(4):
            v += self.new_clicks_function_sigma(bid, i, num_people[i])
        return v

    def cc(self, bid, size = 1):
        return bid*np.random.beta(4.4,bid**0.5, size=size)
    
    def mean_cc(self, bid, classes):
        v = 0
        for i in classes:
            v += bid*(self.costo[i]/(self.costo[i]+bid**0.5))*self.class_distribution[i]
        return v
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
    
    def aggr_return_probability(self, classes):
        aggr_ret = 0
        for c in classes:
            ret_scaled = self.ret[c]*self.class_distribution[c] 
            aggr_ret = np.add(aggr_ret, ret_scaled)
            #aggr_ret += self.config['return_probability'][c]
        return aggr_ret #/ len(classes)
    
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