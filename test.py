"""
The idea of this file is to be able to test the accuracy of the fit.
n_iter sets of size n_data are generated with a Levy distribution of
parameters alpha, beta, mu, sigma.
The 50%, 5% and 95% quantiles of the distributions of the 4 parameters
are returned.
"""

import levy
import numpy as np
from matplotlib import pyplot
#
#
# def get_quantiles(l):
#     n = float(len(l))
#     return l[int(n * 0.5)], l[int(n * 0.05)], l[int(n * 0.95)]
#
# alpha = 1.0
# beta = 0.0
# mu = 0.0
# sigma = 1.0
#
# n_iter = 100
# n_data = 1000
#
# parameters_list = []
# for _ in range(n_iter):
#     data = levy.random(alpha, 0.0, 0.0, 1.0, n_data)
#     parameters = levy.fit_levy(data)
#     parameters_list.append(parameters)
#     if _ % 20 == 0:
#         print _
#
# alphas = sorted([_[0] for _ in parameters_list])
# betas = sorted([_[1] for _ in parameters_list])
# mus = sorted([_[2] for _ in parameters_list])
# sigmas = sorted([_[3] for _ in parameters_list])
#
# print get_quantiles(alphas)
# print get_quantiles(betas)
# print get_quantiles(mus)
# print get_quantiles(sigmas)

par_name = ['alpha', 'beta', 'mu', 'sigma']

with open('test_minimize', 'r') as f:
    res_1 = [list(map(float, x.rstrip('\n').split('\t'))) for x in f]
with open('test_fmin', 'r') as f:
    res_2 = [list(map(float, x.rstrip('\n').split('\t'))) for x in f]

true_values = [1.5, 0.5, 0.0, 1.0]

for i in range(4):
    a1 = [x[i]-true_values[i] for x in res_1]
    a2 = [x[i]-true_values[i] for x in res_2]
    av1 = np.mean(a1)
    av2 = np.mean(a2)
    st1 = np.std(a1)
    st2 = np.std(a2)
    print('Parameter {}'.format(par_name[i]))
    print('minimize: {} +- {}'.format(av1, st1))
    print('fmin: {} +- {}\n'.format(av2, st2))
# bins = np.linspace(1.3, 1.7, 41, endpoint=True)
# pyplot.hist(a1, bins, label='new', alpha=0.5)
# pyplot.hist(a2, bins, label='old', alpha=0.5)
# pyplot.show()


