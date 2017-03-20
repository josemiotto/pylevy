"""
The idea of this file is to be able to test the accuracy of the fit.
n_iter sets of size n_data are generated with a Levy distribution of
parameters alpha, beta, mu, sigma.
The 50%, 5% and 95% quantiles of the distributions of the 4 parameters
are returned.
"""

import levy
import numpy as np
from builtins import range
# from matplotlib import pyplot


def get_quantiles(l):
    n = float(len(l))
    return l[int(n * 0.5)], l[int(n * 0.05)], l[int(n * 0.95)]

alpha = 1.0
beta = 0.0
mu = 0.0
sigma = 1.0

n_iter = 100
n_data = 1000

parameters_list = []
for _ in range(n_iter):
    data = levy.random(alpha, 0.0, 0.0, 1.0, n_data)
    parameters = levy.fit_levy(data)
    parameters_list.append(parameters)
    if _ % 20 == 0:
        print(_)

alphas = sorted([_[0] for _ in parameters_list])
betas = sorted([_[1] for _ in parameters_list])
mus = sorted([_[2] for _ in parameters_list])
sigmas = sorted([_[3] for _ in parameters_list])

print(get_quantiles(alphas))
print(get_quantiles(betas))
print(get_quantiles(mus))
print(get_quantiles(sigmas))
