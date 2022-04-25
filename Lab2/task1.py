from scipy.stats import multivariate_normal, pearsonr, spearmanr
import numpy as np
import statistics as stat
from ellipse import Ellipse
import matplotlib.pyplot as plt


def generate_sample(size, cor_coef):
    return multivariate_normal.rvs(mean=[0., 0.], cov=[[1., cor_coef], [cor_coef, 1.]], size=size)


def generate_mix_sample(size, cor_coef):
    return 0.9 * multivariate_normal.rvs(mean=[0., 0.], cov=[[1., 0.9], [0.9, 1.]], size=size) + \
           0.1 * multivariate_normal.rvs(mean=[0., 0.], cov=[[1., -0.9], [-0.9, 1.]], size=size)


def mean(data):
    return np.mean(data)


def variance(data):
    return stat.variance(data)


def pearson(sample):
    return pearsonr(sample[:, 0], sample[:, 1])[0]


def spearman(sample):
    return spearmanr(sample[:, 0], sample[:, 1])[0]


def quadrant(sample):
    x = sample[:, 0] - np.median(sample[:, 0])
    y = sample[:, 1] - np.median(sample[:, 1])

    n = 0
    for i in range(len(x)):
        if x[i] > 0 and y[i] >= 0 or x[i] < 0 and y[i] <= 0:
            n += 1
        else:
            n -= 1

    return n / len(x)


class Task1:
    def __init__(self):
        self._sizes = [20, 60, 100]
        self._cor_coefs = [0, 0.5, 0.9]
        self._repeat = 1000

    def create_characteristic_table(self, sample_generate_strategy):
        for size in self._sizes:
            table = f"\\begin{{tabular}}{{| c | c | c | c |}} \\hline \n"

            for cor_coef in self._cor_coefs:
                table += f" p = {cor_coef} & $r$ & $r_{{S}}$ & $r_{{Q}}$ \\\\ \\hline \n"

                data = []
                for _ in range(self._repeat):
                    sample = sample_generate_strategy(size, cor_coef)
                    data += [[c_cor(sample) for c_cor in [pearson, spearman, quadrant]]]
                data = np.array(data).T

                res = [list(map(lambda x: x.__round__(3),
                                [
                                    mean(data[i]),
                                    mean(list(map(lambda y: y ** 2, data[i]))),
                                    variance(data[i])
                                ]
                                )) for i in range(3)]

                for i in range(3):
                    if i == 0:
                        table += f" $E(z)$"
                    elif i == 1:
                        table += f" $E(z^2)$"
                    else:
                        table += f" $D(z)$"

                    for x in res:
                        table += f" & {x[i]}"

                    table += f" \\\\ \\hline \n"

            table += f" \\end{{tabular}} \n"

            file = open(
                f"Report/CharacteristicTables/{size}{sample_generate_strategy.__name__}" + ".tex", "w")
            file.write(table)

    def equiprobability_ellipse(self):
        for capacity in self._sizes:
            _, sp = plt.subplots(1, 3, figsize=(16, 6))
            for cor_cov, subplot in zip(self._cor_coefs, sp):
                sample = multivariate_normal.rvs([0, 0], [[1, cor_cov], [cor_cov, 1]], capacity)

                x = sample[:, 0]
                y = sample[:, 1]

                ellipse = Ellipse(0, 0, 1, 1, cor_cov)

                subplot.scatter(x, y)

                x = np.linspace(min(x) - 2, max(x) + 2, 100)
                y = np.linspace(min(y) - 2, max(y) + 2, 100)
                x, y = np.meshgrid(x, y)
                z = ellipse.z(x, y)
                t = ellipse.rad2(sample)
                subplot.contour(x, y, z, [ellipse.rad2(sample)])

                title = f"n = {capacity} rho = {cor_cov}"

                subplot.set_title(title)
                subplot.set_xlabel("X")
                subplot.set_ylabel("Y")

            plt.savefig(f"Report/images/{capacity}" + "ellipse" + f".png")

    def run(self):
        self.create_characteristic_table(generate_sample)
        self.equiprobability_ellipse()
        self._cor_coefs = [0.9]
        self.create_characteristic_table(generate_mix_sample)
