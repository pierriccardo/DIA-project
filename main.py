import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


print(config["seed"])
print(config["params"]["avg_cc"])