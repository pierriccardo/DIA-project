from numpy import split
from learners import Learner, TS_Learner
import logging
import itertools

class Context:

    def __init__(self, id, classes, learner):

        self.id = id
        self.classes = classes
        self.learner = learner
    
    def has_feature(self, f):
        f1, f2 = False, False
        for c in self.classes:
            if f[0] in c: f1 = True
            if f[1] in c: f2 = True
        return f1 and f2


class ContextGenerator():

    def __init__(self, obs, n_arms, classes, features, candidates):
        self.n_arms = n_arms
        self.candidates = candidates 
        self.obs = obs
        self.classes = classes # [Y, I], [Y, D], [A, I], [A, D]
        self.features = features # [Y, A], [I, D]
        self.contexts = []
        self.split_color_matrices = []  
        self.current_id = 0


        ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.candidates)
        init_context = Context(0, self.classes, ts_learner)
        self.contexts.append(init_context)

    def init_context(self):
        self.current_id = 0
        self.contexts = []
        ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.candidates)
        # we train the learner since observations may be not empty at the 
        # beginning e.g. if some delay are applied
        self.train_learner(ts_learner, self.obs)
        init_context = Context(0, self.classes, ts_learner)
        self.contexts.append(init_context)

    def train_learner(self, learner, obs):
        for o in obs:
            learner.update(o[1], o[2])
    
    def train_new_learner(self, obs):
        ts_learner = TS_Learner(n_arms=self.n_arms, candidates=self.candidates)
        for o in obs:
            ts_learner.update(o[1], o[2])
        return ts_learner
            
    def update(self, o):
        self.obs.append(o)
        for c in self.contexts:
            if o[0] in c.classes:
                c.learner.update(o[1], o[2])

    def pull_arm(self):
        # ad ogni round (giorno) vengono tirati gli arm di ogni context
        # questi vengono salvati in pulled_arms nel seguente formato:
        # [classi_del_context, pulled_arm], ...
        # così che ad ogni nuova persona che arriva verrà selezionato
        # l'arm che ha tra le sue classi quella del cliente 
        pulled_arms = [] 

        for c in self.contexts:
            pulled_arm, _ = c.learner.pull_arm()
            pulled_arms.append([c.classes, pulled_arm])
        return pulled_arms      

    def extract_obs(self, classes):
        # extract obs which belongs to the classes passed as argument
        return [o for o in self.obs if o[0] in classes]

    def evaluate_split(self, context, feature):
        classes_1, classes_2 = self.separate_classes(context.classes, feature)
        obs_1 = self.extract_obs(classes_1)
        obs_2 = self.extract_obs(classes_2)

        learner_1 = self.train_new_learner(obs_1)
        learner_2 = self.train_new_learner(obs_2)

        if len(obs_1) + len(obs_2) > 0:
            p_1 = len(obs_1) / (len(obs_1) + len(obs_2))
            p_2 = len(obs_2) / (len(obs_1) + len(obs_2)) 
        else:
            p_1 = p_2 = 0
        
        mu_1 = learner_1.expected_value_lower_bound()
        mu_2 = learner_2.expected_value_lower_bound()
        mu_0 = context.learner.expected_value_lower_bound()
        
        logging.debug(f'Context.split_evaluation() -> evaluating split context id {context.id} | feature {feature}')

        msg = f'Context.split_evaluation() -> id=c{context.id} p1 = {p_1:.2f}|p2 = {p_2:.2f}|mu1 = {mu_1:.2f}| mu2 = {mu_2:.2f}|mu0 = {mu_0:.2f}|'
        logging.debug(f'{msg}')

        return (p_1 * mu_1 + p_2 * mu_2) - mu_0
    
    def generate(self):

        for c in self.contexts:
            split_conditions = []

            for f in self.features: # [Y, A], [I, D]
                if c.has_feature(f): 
                    split_cond = self.evaluate_split(c, f) 
                    if split_cond >= 0:
                        split_conditions.append([f, split_cond]) # [[Y, A], 0.3345]
            
            if len(split_conditions) > 0:
            
                best_split = split_conditions[0]
                for e in split_conditions:
                    if e[1] > best_split[1]:
                        best_split = e
                best_feature = best_split[0]
                
                classes_1, classes_2 = self.separate_classes(c.classes, best_feature)
                obs_1 = self.extract_obs(classes_1)
                obs_2 = self.extract_obs(classes_2)

                learner_1 = self.train_new_learner(obs_1)
                learner_2 = self.train_new_learner(obs_2)

                self.current_id += 1
                c1 = Context(self.current_id, classes_1, learner_1)
                self.current_id += 1
                c2 = Context(self.current_id, classes_2, learner_2)
                        
                self.contexts.append(c1)
                self.contexts.append(c2)
                self.contexts.remove(c)

                logging.debug(f'ContextGenerator.generate() c_{c.id} splitted in: c_{c1.id}({classes_1}),c_{c2.id}({classes_2})')                
            
    def separate_classes(self, classes, feature):
        classes1, classes2 = [], []
        for c in classes:
            if feature[0] in c: classes1.append(c)
            if feature[1] in c: classes2.append(c)
        return classes1, classes2

    def expected_value_arm(self, user_class, pulled_arm):
        for c in self.contexts:
            if user_class in c.classes:
                return c.learner.expected_value(pulled_arm)
    
    def update_reward_per_experiments(self):
        for c in self.contexts:
            c.reward_per_experiments.append(c.learner.collected_rewards)

    def get_collected_reward(self):
        reward = 0
        for c in self.contexts:
            reward += c.learner.collected_rewards
        return reward
    
    # function to plot the contexts 
    def add_split_matrix(self):
        color = 0
        class_colors = {}
        for c in self.contexts:
            ctx_classes = [c[0]+c[1] for c in c.classes]
            for f in itertools.product(self.features[0], self.features[1]):
                if f[0]+f[1] in ctx_classes:
                    class_colors[f[0]+f[1]] = {}
                    class_colors[f[0]+f[1]] = color
            color += 1
        self.split_color_matrices.append(class_colors)
    
    def get_split_matrices(self):
        return self.split_color_matrices
                
    
