import yaml
from pricing import *
import numpy as np

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)



a = [1,2,3]
b = [3,2,5]
c = np.add(a,b)
print(np.add(a,b))
print(np.divide(c, 2))





