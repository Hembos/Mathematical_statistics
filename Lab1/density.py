import numpy as np
from scipy.special import factorial
import scipy.stats as stats


class Density:
    @staticmethod
    def normal(x: list, mu=0, sigma=1):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

    @staticmethod
    def cauchy(x: list):
        return 1 / (np.pi * (x**2 + 1))

    @staticmethod
    def laplace(x: list, loc=0, scale=1/np.sqrt(2)):
        return np.exp(-abs(x - loc) / scale) / (2. * scale)

    @staticmethod
    def poisson(k: list, lam=10):
        return lam**k * np.exp(-lam) / factorial(k)

    @staticmethod
    def uniform(x: list, low=-np.sqrt(3), high=np.sqrt(3)):
        pdf = []
        for each_x in x:
            if np.fabs(each_x) <= high:
                pdf.append(1 / (high - low))
            else:
                pdf.append(0)
        return pdf


class CumulativeDensity:
    @staticmethod
    def normal(x: list, mu=0, sigma=1):
        return stats.norm(mu, sigma).cdf(x)

    @staticmethod
    def cauchy(x: list):
        return stats.cauchy(loc=0, scale=1).cdf(x)

    @staticmethod
    def laplace(x: list, loc=0, scale=1/np.sqrt(2)):
        return stats.laplace(loc=loc, scale=scale).cdf(x)

    @staticmethod
    def poisson(k: list, lam=10):
        return stats.poisson(lam).cdf(k)

    @staticmethod
    def uniform(x: list, low=-np.sqrt(3), high=np.sqrt(3)):
        return stats.uniform(low, high).cdf(x)
