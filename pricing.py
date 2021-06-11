import numpy as np
import logging

class PersonGenerator():

    def __init__(self, classes, probabilities):
        self.classes = classes
        self.probabilities = probabilities
  
    def generate_person(self):
        # generate 1 person belonging to a random class
        
        n = np.random.choice(4, size=1, p=self.probabilities)[0]
        logging.debug(f'PersonGenerator.generate_person() class: {n}|labels: {self.classes[n]}')

        return self.classes[n]

    def generate_people(self, num=8):
        # TODO: make it random ? or pass as argument a random generated number?
        # generate a random number of people
        people = []
        for _ in range(num):
            people.append(self.generate_person())

        return people


