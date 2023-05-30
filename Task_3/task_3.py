import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

def f(x):
    return np.exp(-np.abs(x)) / 2 / (1 - math.e ** (-1))

def F(x):
    if x < 0:
        return (np.exp(-x) - np.exp(-1)) / (2 * (1 - np.exp(-1)))
    else:
        return 1 / 2 + (1 - np.exp(-x)) / (2 * (1 - np.exp(-1)))

def inv_F(x):
    if x < 1 / 2:
        return -np.log(2 * (1 - np.exp(-1)) * x + np.exp(-1))
    else:
        return -np.log(1 - 2 * (1 - np.exp(-1)) * (x - 1 / 2))


class density_distrib(rv_continuous):
    def __init__(self, f):
        super().__init__(a=-1, b=1)
        self._pdf = f

class const_distrib(rv_continuous):
    def __init__(self):
        super().__init__(a=0, b=1)
    def _pdf(self, x, *args):
        return 1

class inverse_distrib:

    def __init__(self, invf):
        self.constd = const_distrib()
        self.invf = invf

    def rvs(self):
        return self.invf(self.constd.rvs())

# легко вычисляемое приближение f
c = 1 - 1 / (2 * (1 - math.e ** -1))
b = 1 / (1 - math.e ** -1) - 1
M = 1 / (2 * (1 - math.e ** -1) * math.e) / c
def g(x):
    if x < 0:
        return (x + 1) * b + c
    else:
        return c + (1 - x) * b

class reject_distrib:

    def __init__(self, M, f, g, g_distr):
        self.f = f
        self.g = g
        self.D = g_distr
        self.M = M
        self.U = const_distrib()

    def rvs(self):
        while True:
            u = self.U.rvs()
            x = self.D.rvs()
            if u < self.f(x) / (self.M * self.g(x)):
                return x

def test(tm, func, num):

    sum_tm = 0

    way = []
    i = 0
    while True:
        i = i + 1
        start = time.time()
        for j in range(i):
            func.rvs()
        end = time.time()
        sum_tm += end - start
        if sum_tm > tm:
            break
        way.append([i, end - start])
    way = np.array(way)
    print('test', num, ':', i)
    plt.figure()
    plt.plot(way[:, 0], way[:, 1], '-', color = 'green')
    plt.show()

first = density_distrib(f)
second = inverse_distrib(inv_F)
third = reject_distrib(M, f, g, density_distrib(g))

tm = 1

print('Tests result in', tm, 'seconds:')
test(tm, first, 1)
test(tm, second, 2)
test(tm, third, 3)