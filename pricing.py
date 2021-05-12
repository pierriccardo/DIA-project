import numpy as np
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

def conv_rate(x, a=1, b=1, c=1):
    return ((c*x) ** a) * np.exp(-b * c * x)

def cost_per_click(bid, alpha):
    beta = np.sqrt(bid)
    return bid * np.random.beta(alpha, beta, 1)

def return_probability(_lambda, size=1):
    samples = np.random.poisson(_lambda, size=size)
    return samples if size > 1 else samples[0]

def new_clicks(bid, Na=10000, p0=0.01, cc=0.44):
    p = 1-(cc/(2*bid))
    mean = Na*p*p0
    sd = (Na*p*p0*(1-p0))**0.5
    return np.random.normal(mean,sd)

def aggregated_new_cliks(bid, Na=10000, cc=0.44):
    return config["frequencies"]["class1"]*new_clicks(bid, Na, new_clicks["class1"][1])+config["frequencies"]["class2"]*new_clicks(bid, Na, new_clicks["class2"][1])+config["frequencies"]["class3"]*new_clicks(bid, Na, new_clicks["class3"][1])+config["frequencies"]["class4"]*new_clicks(bid, Na, new_clicks["class4"][1])


# def aggregated_conv_rate(x): BISOGNA CAMBIARE CONFIG PER FARLA ANDARE
#    return config["frequencies"]["class1"]*conv_rate(x,conv_rate["class1"][0],conv_rate["class1"][1],conv_rate["class1"][2])+config["frequencies"]["class2"]*conv_rate(x,conv_rate["class2"][0],conv_rate["class2"][1],conv_rate["class2"][2])+frequencies["class3"]*conv_rate(x,conv_rate["class3"][0],conv_rate["class3"][1],conv_rate["class3"][2])+frequencies["class4"]*conv_rate(x,conv_rate["class4"][0],conv_rate["class4"][1],conv_rate["class4"][2])

def aggregated_return_proba(bid):
    return config["frequencies"]["class1"]*return_probability[0]+config["frequencies"]["class2"]*return_probability[1]+config["frequencies"]["class3"]*return_probability[2]+config["frequencies"]["class4"]*return_probability[3]

def aggregated_conv_rates(prices, num_classes=4, num_candidates=10):    

    aggr_conv_rate = np.zeros(num_candidates)

    for user_class in config["classes"]:
        a,b,c = tuple(config["conv_rate"][user_class])
        conv_rates = []

        for price in prices:
            conv_rates = [i for i in conv_rate(price, a, b, c)]
        
        aggr_conv_rate = np.add(aggr_conv_rate, conv_rates)
    
    return np.divide(aggr_conv_rate, num_classes)
        


        

