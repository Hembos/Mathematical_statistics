import numpy as np


class Distribution:

    @staticmethod
    def normal(size):
        return np.random.normal(0, 1, size)

    @staticmethod
    def cauchy(size):
        return np.random.standard_cauchy(size)

    @staticmethod
    def laplace(size):
        return np.random.laplace(0, np.sqrt(2), size)

    @staticmethod
    def poisson(size):
        return np.random.poisson(10, size)

    @staticmethod
    def uniform(size):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), size)