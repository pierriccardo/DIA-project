import yaml
from pricing import *

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


print(config["seed"])
print(config["conv_rate"]['young']['interested'])


bid=0.5
x=5.0

print(aggregated_new_cliks(bid))

# print(aggregated_conv_rate(x))

print(aggregated_return_proba(bid))
 