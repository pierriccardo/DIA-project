import numpy as np
from learners import Learner, TS_Learner
from utils import *
import logging

class Context():
    """
    A context is defined by: 
    - an ID to identify it 
    - a learner algorithm
    - one or more subspaces of features 
    subspace is expressed as an array of tuple [('Y', 'I'),...]
    """

    # TODO: domanda, se io ho 3 context, al round x pullo i 3 context, quindi devo 
    # mandare all'arm pullato (il prezzo) tutte le persone appartenenti alla classe del context?

    def __init__(self, id, learner, classes, obs=[]):
        """
        id: is the context identifier
        learner: learner associated with the context
        classes: classes of the context e.g. [["Y", "I"], ["Y", "D"]]
        obs: observations log from environment e.g. [[user_class, pulled_arm, reward], ...] 
        """
        
        self.id = id
        self.learner = learner
        self.obs = obs
        self.context_obs = []
        self.classes = classes

    def update(self, new_obs):
        # updates context learner
        self.learner.update(new_obs[1], new_obs[2]) # pulled_arm, reward

        # update context observations
        self.obs.append(new_obs)

        logging.info(f'Context.update()|id: {self.id}|class: {new_obs[0]}|pulled_arm: {new_obs[1]}|reward: {new_obs[2]}')

    def train_sub_learner(self, obs):
        # train a new learner, using observations
        learner = TS_Learner(self.learner.n_arms, self.learner.candidates)
        
        # train the learner
        for o in obs:
            learner.update(o[1], o[2])

        return learner
    
    def train_learner(self):
        for o in self.obs:
            # we train the learner with observations which have
            # user_class belonging to this context classes

            # e.g. 
            # context classes = [["Y", "I"], ["Y", "D"]] 
            # obs = (["Y", "I"], pulled_arm, reward) -> accepted
            # obs = (["A", "I"], pulled_arm, reward) -> refused
           
            for c in self.classes:
                for f in o[0]:
                    if f in c:
                        self.learner.update(o[1], o[2])
                        self.context_obs.append(o)


    def split_evaluation(self, feature):
        """
        return the value obtained by splitting 
        for the feature

        feature is an array of type ["A", "B"]
        it identify a feature 
        """

        # split the complete log of observations
        # in two logs one with feature "A", one for "B"
        obs_1 = self.retrieve_obs(feature[0]) 
        obs_2 = self.retrieve_obs(feature[1]) 

        learner_1 = self.train_sub_learner(obs_1)
        learner_2 = self.train_sub_learner(obs_2)

        p_1 = len(obs_1) / len(self.obs) if len(self.obs) != 0 else 0
        p_2 = len(obs_2) / len(self.obs) if len(self.obs) != 0 else 0

        
        mu_1 = learner_1.expected_value_lower_bound(len(obs_1) + 1)
        mu_2 = learner_2.expected_value_lower_bound(len(obs_2) + 1)

        mu_1 = mu_1 if mu_1 is not np.nan else 0
        mu_2 = mu_2 if mu_2 is not np.nan else 0    
        

        mu_0 = self.learner.expected_value_lower_bound(len(self.context_obs) + 1)

        msg = f'[p1 = {p_1}|p2 = {p_2}|mu0 = {mu_0}|mu1 = {mu_1}| mu2 = {mu_2}]'
        logging.info(f'Context.split_evaluation() {msg}')

        return p_1 * mu_1 + p_2 * mu_2 >= mu_0, learner_1, learner_2

    def retrieve_obs(self, feature):
        selected_obs = []
        for o in self.obs:
            if feature in o[0]:
                selected_obs.append(o)
        return selected_obs 

    def has_features(self, features):
        # check if the context has in its classes
        # both features passed as arguments
        # used to see if the context has already
        # splitted for that feature
        f1 = False
        f2 = False
        for c in self.classes:
            if features[0] in c: f1 = True
            if features[1] in c: f2 = True

        return f1 and f2 

class ContextGenerator():

    def __init__(self, n_arms, classes, features, candidates, obs):
        
        self.n_arms = n_arms 
        self.classes = classes 
        self.features = features 
        self.candidates = candidates
        self.obs = obs

        self.contexts = []

        # generate first context with all the classes
        ts_learner = TS_Learner(n_arms, candidates)
        init_context = Context(0, ts_learner, classes, self.obs)
        init_context.train_learner()

        logging.info(f'ContextGenerator.__init__() created context c_{init_context.id}->{init_context.classes}')
        
        self.contexts.append(init_context)

    def generate(self):

        for f in self.features:
            for c in self.contexts:

                # TODO: check if context has the two features already splitted
                if not c.has_features(f):
                    continue

                # evaluate wheter it is worth to 
                # split the context for the feature "f" 
                 
                if c.split_evaluation(f):

                    # if split_evaluation is true, then is worth splitting
                    # so we create the 2 contexts
                    learner_1 = TS_Learner(self.n_arms, self.candidates)
                    learner_2 = TS_Learner(self.n_arms, self.candidates)

                    # split the classes of context c
                    classes_1 = [c for c in c.classes if f[0] in c]
                    classes_2 = [c for c in c.classes if f[1] in c]

                    c1 = Context(c.id+1, learner_1, classes_1)
                    c1.train_learner()

                    c2 = Context(c.id+2, learner_2, classes_2)
                    c2.train_learner() 

                    logging.info(f'ContextGenerator.generate() created context c_{c.id+1}->{classes_1},c_{c.id+2}->{classes_2}')

                    self.contexts.append(c1)
                    self.contexts.append(c2)
    
    def update(self, new_obs):
        self.obs.append(new_obs)

        for c in self.contexts:
            if new_obs[0] in c.classes:
                c.update(new_obs)

    def find_context(self, user_class):
        for c in self.contexts:
            if user_class in c.classes:
                return c
        logging.error("ContextGenerator.find_context() no context with class {user_class}")
    
    def pull_arm(self, user_class):

        # find the correct context and pull the arm
        context = self.find_context(user_class)
        return context.learner.pull_arm()
    
    def expected_value_arm(self, user_class, pulled_arm):

        context = self.find_context(user_class)
        return context.learner.expected_value(pulled_arm)





    

