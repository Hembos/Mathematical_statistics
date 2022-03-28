from numpy import random
from numpy import sqrt


class Distribution:
    @staticmethod
    def normal(size: int, mu=0, sigma=1):
        return random.normal(loc=mu, scale=sigma, size=size)

    @staticmethod
    def cauchy(size: int):
        return random.standard_cauchy(size)

    @staticmethod
    def laplace(size: int, loc=0, scale=1/sqrt(2)):
        return random.laplace(loc=loc, scale=scale, size=size)

    @staticmethod
    def poisson(size: int, lam=10):
        return random.poisson(lam=lam, size=size)

    @staticmethod
    def uniform(size: int, low=-sqrt(3), high=sqrt(3)):
        return random.uniform(low=low, high=high, size=size)
