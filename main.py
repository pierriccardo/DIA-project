import yaml
from pricing import *
import numpy as np

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


print(config["seed"])
print(config["conv_rate"]['young']['interested'])






