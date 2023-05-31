import math
import time
import numpy as np
from scipy.stats import poisson

def test(eps, delta, lmb, n):
    gen = poisson(lmb)
    am = 0
    m = 100
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += gen.rvs()
        sum /= n
        #print(sum, lmb)
        if abs(sum - lmb) <= eps:
            am += 1
    am /= m
    if am >= 1 - delta:
        print('Experiment successful:', am, '>', 1 - delta)
    else:
        print('Experiment failed:', am, 'must be greater then', 1 - delta)

def Phi(x):
    return (1 + math.erf(-math.sqrt(x) * 0.01 / 10))

def get_x_CPT():
    i = 0
    while True:
        i += 1
        if Phi(i) <= 0.05:
            return i

n_chebyshev = 2000000
n_CPT = get_x_CPT()

print(n_chebyshev, n_CPT)

test(0.01, 0.05, 10, n_chebyshev)
test(0.01, 0.05, 10, n_CPT)