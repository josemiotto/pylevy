"""
The idea of this file is to be able to test the accuracy of the fit.
n_iter sets of size n_data are generated with a Levy distribution of
parameters alpha, beta, mu, sigma.
The 50%, 5% and 95% quantiles of the distributions of the 4 parameters
are returned.
"""
import numpy as np
import levy


def get_quantiles(l):
    n = float(len(l))
    return l[int(n * 0.5)], l[int(n * 0.05)], l[int(n * 0.95)]


np.random.seed(0)

alpha = 0.5 + 1.5 * np.random.rand()
beta = -1 + 2 * np.random.rand()
mu = 0.0
sigma = 1.0

n_iter = 100
n_data = 1000

parameters_list = []
for _ in range(n_iter):
    data = levy.random(alpha, beta, mu, sigma, n_data)
    parameters = levy.fit_levy(data)
    print(parameters)
    parameters_list.append(parameters)
    if _ % 20 == 0:
        print(_)

alphas = sorted([_[0].x[0] for _ in parameters_list])
betas = sorted([_[0].x[1] for _ in parameters_list])
mus = sorted([_[0].x[2] for _ in parameters_list])
sigmas = sorted([_[0].x[3] for _ in parameters_list])

print(get_quantiles(alphas))
print(get_quantiles(betas))
print(get_quantiles(mus))
print(get_quantiles(sigmas))
