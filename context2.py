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
        self.context_color_matrix = []  

        self._init_context()

    def _init_context(self):
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

    def pull_arm(self, user_class):
        context = self.find_context(user_class)
        return context.learner.pull_arm()

    def find_context(self, user_class):
        for c in self.contexts:
            if user_class in c.classes:
                return c
        logging.error("ContextGenerator.find_context() no context with class {user_class}")

    def extract_obs(self, classes):
        # extract obs which belongs to the classes passed as argument
        return [o for o in self.obs if o[0] in classes]
    
    def generate(self):
        #for c in self.contexts:
        #    mu_0 = c.learner.expected_value_lower_bound()

        for f in self.features: # [Y, A], [I, D]
            for c in self.contexts:
                if c.has_feature(f):

                    classes_1, classes_2 = self.separate_classes(c.classes, f)
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
                    mu_0 = c.learner.expected_value_lower_bound()

                    #mu_1 = mu_1 if mu_1 > 0 else 0
                    #mu_2 = mu_2 if mu_2 > 0 else 0
                    #mu_0 = mu_0 if mu_0 > 0 else 0

                    if p_1 * mu_1 + p_2 * mu_2 >= mu_0: # do the split

                        msg = f'Context.split_evaluation() -> p1 = {p_1}|p2 = {p_2}|mu0 = {mu_0}|mu1 = {mu_1}| mu2 = {mu_2}'
                        logging.debug(f'Context.split_evaluation() {msg}')

                        c1 = Context(c.id + 1, classes_1, learner_1)
                        c2 = Context(c.id + 2, classes_2, learner_2)
                    
                        self.contexts.append(c1)
                        self.contexts.append(c2)
                        self.contexts.remove(c)
                        self.add_context_color_matrix()

                        return True
        return False
                        
    def separate_classes(self, classes, feature):
        classes1, classes2 = [], []
        for c in classes:
            if feature[0] in c: classes1.append(c)
            if feature[1] in c: classes2.append(c)
        return classes1, classes2

    def expected_value_arm(self, user_class, pulled_arm):

        context = self.find_context(user_class)
        return context.learner.expected_value(pulled_arm)
    
    def update_reward_per_experiments(self):
        for c in self.contexts:
            c.reward_per_experiments.append(c.learner.collected_rewards)

    def get_collected_reward(self):
        reward = 0
        for c in self.contexts:
            reward += c.learner.collected_rewards
        return reward

    # function to plot the contexts 
    def add_context_color_matrix(self):
        color = 0
        class_colors = {}
        for c in self.contexts:
            ctx_classes = [c[0]+c[1] for c in c.classes]
            for f in itertools.product(self.features[0], self.features[1]):
                if f[0]+f[1] in ctx_classes:
                    class_colors[f[0]+f[1]] = {}
                    class_colors[f[0]+f[1]] = color
            color += 1
        class_colors['obs'] = len(self.obs)
        self.context_color_matrix.append(class_colors)
    
    def get_context_color_matrices(self):
        return self.context_color_matrix
                
    
