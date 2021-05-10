import yaml
from pricing_enviroment import PricingEnvironment

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


print(config["seed"])
print(config["conv_rate"]['young']['interested'])

prices = config["pricing"]
n_arms = len(prices)

PricingEnvironment(n_arms,prices)