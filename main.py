import yaml
import numpy as np
from utils import *

from pricing import PersonGenerator

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)



#a = [1,2,3]
#b = [3,2,5]
#c = np.add(a,b)
#print(np.add(a,b))
#print(np.divide(c, 2))

pg = PersonGenerator(None, config["class_distribution"])

for _ in range(10):
    print(pg.generate_person())






