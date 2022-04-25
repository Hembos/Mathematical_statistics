from distribution import Distribution
import numpy as np
import scipy.stats as stats


class Task4:
    def run(self):
        sizes = [20, 100]
        alpha = 0.05
        latex1 = f"\\begin{{tabular}}{{| c | c | c |}} \\\\ \\hline \n"
        latex2 = f"\\begin{{tabular}}{{| c | c | c |}} \\\\ \\hline \n"
        for size in sizes:
            sample = Distribution.normal(size)
            mu = np.mean(sample)
            sigma = np.std(sample)

            mu_classic = [mu - sigma * (stats.t.ppf(1 - alpha / 2, size - 1)) / np.sqrt(size - 1),
                          mu + sigma * (stats.t.ppf(1 - alpha / 2, size - 1)) / np.sqrt(size - 1)]
            sigma_classic = [sigma * np.sqrt(size) / np.sqrt(stats.chi2.ppf(1 - alpha / 2, size - 1)),
                             sigma * np.sqrt(size) / np.sqrt(stats.chi2.ppf(alpha / 2, size - 1))]

            mu_asymptotic = [mu - sigma * stats.norm.ppf(1 - alpha / 2) / np.sqrt(size),
                             mu + sigma * stats.norm.ppf(1 - alpha / 2) / np.sqrt(size)]
            sigma_asymptotic = [sigma / np.sqrt(
                1 + stats.norm.ppf(1 - alpha / 2) * np.sqrt((stats.moment(sample, 4) / sigma ** 4 + 2) / size)),
                                sigma / np.sqrt(1 - stats.norm.ppf(1 - alpha / 2) * np.sqrt(
                                    (stats.moment(sample, 4) / sigma ** 4 + 2) / size))]

            latex1 += f"n = {size} & $m$ & $\\sigma$ \\\\ \\hline \n"
            latex1 += f" & ${mu_classic[0].__round__(3)} < m < {mu_classic[1].__round__(3)}$ & ${sigma_classic[0].__round__(3)} < \\sigma < {sigma_classic[1].__round__(3)}$ \\\\ \\hline \n"
            latex1 += f" & & \\\\ \\hline \n"

            latex2 += f"n = {size} & $m$ & $\\sigma$ \\\\ \\hline \n"
            latex2 += f" & ${mu_asymptotic[0].__round__(3)} < m < {mu_asymptotic[1].__round__(3)}$ & ${sigma_asymptotic[0].__round__(3)} < \\sigma < {sigma_asymptotic[1].__round__(3)}$ \\\\ \\hline \n"
            latex2 += f" & & \\\\ \\hline \n"

        latex1 += f" \\end{{tabular}} \n"
        latex2 += f" \\end{{tabular}} \n"

        file = open(
            f"Report/Confidence intervals/classic.tex", "w")
        file.write(latex1)
        file.close()
        file = open(
            f"Report/Confidence intervals/asymptotic.tex", "w")
        file.write(latex2)
