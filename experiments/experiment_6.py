import numpy as np

class Experiment6():
    
    def __init__(self):
        self.cm = ConfigManager()

        # pricing
        self.prices = self.cm.prices # candidates

        self.p = [.12, .3, .1, .5, .07, .43, .03, .02, .34, .06] # probabilities (conv rate)
        self.n_arms = len(self.prices) #p = cm.aggr_conv_rates()
        self.opt_pricing = np.max(np.multiply(self.p, self.prices)) 

        

        # bidding 
        self.bids = np.array(self.cm.bids)
        self.sigma = 10
        self.p = [.1, .03, .34, .28, .12, .19, .05, .56, .26, .35]# cm.aggr_conv_rates()
        
        self.means = self.cm.new_clicks(self.bids)
        self.opt = np.max(self.means * (self.opt_pricing - self.bids))
    
        self.gpts_reward_per_experiment = []
        self.p_arms = []

        self.ts_reward_per_experiments = []

        self.T = 365 # number of days
        self.n_experiments = 100
