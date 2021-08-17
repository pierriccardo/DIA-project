import numpy as np
import logging
from configmanager import ConfigManager

class PersonGenerator():

    def __init__(self):
        self.config = ConfigManager()

        self.classes = self.config.get_classes()
        self.distribution = self.config.class_distribution
  
    def generate_person(self):
        # generate 1 person belonging to a random class
        
        n = np.random.choice(4, size=1, p=self.distribution)[0]
        logging.debug(f'PersonGenerator.generate_person() class: {n}|labels: {self.classes[n]}')

        return n, self.classes[n]

    def generate_people(self):

        num = np.random.binomial(n=500, p=0.6)
        people = []
        for _ in range(num):
            people.append(self.generate_person())

        return people


