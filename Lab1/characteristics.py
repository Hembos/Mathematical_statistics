import numpy as np
import math


class Characteristic:
    @staticmethod
    def sample_mean(x):
        return sum(x) / len(x)
    
    @staticmethod
    def sample_median(x):
        num_elements = len(x)
        l = num_elements // 2
        if num_elements % 2 == 0:
            return (x[l] + x[l + 1]) / 2
        else:
            return x[l + 1]
        
    @staticmethod
    def z_R(x):
        return (x[0] + x[len(x) - 1]) / 2
    
    @staticmethod
    def z_Q(x):
        return (np.quantile(x, 1 / 4) + np.quantile(x, 3 / 4)) / 2
    
    @staticmethod
    def truncated_mean(x):
        num_elements = len(x)
        r = num_elements // 4
        return sum(x[r : num_elements - r - 1]) / (num_elements - 2 * r)
    
    @staticmethod
    def dispersion(self, x):
        s_mean = self.sample_mean(x)
        n = len(x)
        s = 0
        for i in range(1, n):
            s += (x[i] - s_mean) ** 2

        return s / n

    @staticmethod
    def variance(x):
        x = np.array(x)
        mean = Characteristic.sample_mean(x)
        return Characteristic.sample_mean(x * x) - mean * mean

    @staticmethod
    def correct_digits(vrnc: float):
        return max(0, round(-math.log10(abs(vrnc))))