import math

import numpy as np
from distribution import Distribution
from math import ceil
import  scipy.stats as stats

def get_k(size):
    return ceil(1.72 * size**(1/3))

class Task3:
    def create_table(self, size, distribution):
        alpha = 0.05
        p = 1 - alpha
        sample = distribution(size)
        mu = np.mean(sample)
        sigma = np.std(sample)
        k = get_k(size)
        chi_2 = stats.chi2.ppf(p, k - 1)

        latex = f"\\begin{{table}}[H]\n" \
                f"\\label{{tabular:timesandtenses}}\n" \
                f"\\begin{{center}}\n"
        latex += f"$\\hat{{\\mu}} \\approx {mu.round(3)}$, $\\hat{{\\sigma}} \\approx {sigma.round(3)}$\\\\ \n"
        latex += f"\\begin{{itemize}}\n" \
                 f" \\item {{Количество промежутков $k={k}$}}\n" \
                 f" \\item {{Уровень значимости $\\alpha={0.05}$}}\n" \
                 f" \\item {{$\\chi^2={chi_2.round(3)}$}}\n" \
                 f"\\end{{itemize}}\n"

        latex += f"\\begin{{tabular}}{{| c | c | c | c | c | c | c |}} \\hline \n"
        latex += f" i & Границы & $n_i$ & $p_i$ & $np_i$ & $n_i-np_i$ & $\\frac{{(n_i-np_i)^2}}{{np_i}}$ \\\\ \\hline \n"

        limits = np.linspace(-1, 1, num=k-1)
        limits = np.insert(limits, 0, -np.inf)
        limits = np.append(limits, np.inf)
        s = [0., 0., 0., 0., 0.]
        for i in range(k):
            latex += f"{i + 1}" \
                     f" & {[limits[i].__round__(3), limits[i + 1].__round__(3)]}"
            n = len(list(filter(lambda y: limits[i] <= y < limits[i + 1], sample)))
            p = stats.norm.cdf(limits[i + 1]) - stats.norm.cdf(limits[i])

            tmp = [n, p, size * p, n - size * p, math.pow(n - size * p, 2) / (size * p)]
            j = 0
            for x in tmp:
                s[j] += x
                j += 1
                latex += f" & {x.__round__(3)}"

            latex += f"\\\\ \\hline \n"

        latex += "$\sum$ & -"
        for x in s:
            latex += f" & {x.__round__(3)}"

        latex += f"\\\\ \\hline \n"
        latex += f" \\end{{tabular}} \n"
        latex += f"\\end{{center}}\n"

        return latex

    def run(self):
        distr = [Distribution.normal, Distribution.uniform, Distribution.laplace]
        sizes = [100, 20, 20]
        for d, s in zip(distr, sizes):
            latex = self.create_table(s, d)
            file = open(
                f"Report/chi_tables/{d.__name__}" + ".tex", "w")
            file.write(latex)


