import yaml
import numpy as np

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


#print(config["seed"])
#print(config["conv_rate"]['young']['interested'])

ts = np.ndarray([4,5])
print(2 - ts)