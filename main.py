import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


print(config["seed"])
print(config["conv_rate"]["young"]["passionate"])
print(config["conv_rate"]["young"]["passionate"][0])