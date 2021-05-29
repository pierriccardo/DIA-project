import numpy as np

class PersonGenerator():

    def __init__(self, features, probabilities):
        self.features = features
        self.probabilities = probabilities
  
    def generate_person(self):
        
        n = np.random.choice(4, size=1, p=self.probabilities)[0]

        return a[n]

