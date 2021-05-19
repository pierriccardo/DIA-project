import yaml
from pricing import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm



with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


# print(config["seed"])
# print(config["conv_rate"]['young']['interested'])


x = [[0.73684211]]
y = [814.97811682]
alpha = 10.0
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
gp = GaussianProcessRegressor(kernel = kernel, alpha = alpha**2, normalize_y = True, n_restarts_optimizer = 9)
gp.fit(x,y)
print(x)



