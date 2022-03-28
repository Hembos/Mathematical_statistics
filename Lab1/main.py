import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

from density import Density, CumulativeDensity
from distribution import Distribution
from characteristics import Characteristic


def plot_distribution_histograms():
    def plot(func: dict, sizes: list, density_smoothness=100, bins_num=30):
        fig, axs = plt.subplots(
            1, len(sizes), constrained_layout=True, figsize=(12, 3))
        fig.suptitle(func["name"])
        for i in range(len(sizes)):
            distribution = func["distribution"](sizes[i])
            density_x = np.linspace(min(distribution), max(
                distribution), density_smoothness)
            density_y = func["density"](density_x)

            axs[i].hist(distribution, bins=bins_num, density=True)
            axs[i].set_title(f"n={sizes[i]}")
            axs[i].set(xlabel="DistributionNumbers", ylabel="Density")
            axs[i].plot(density_x, density_y, linewidth=2, color='r')

        plt.savefig("Report/pictures/DistributionHistograms/" + func["name"] + ".png")

    size = [10, 50, 1000]
    funcs = [
        {'name': 'Normal', 'distribution': Distribution.normal,
            'density': Density.normal},
        {'name': 'Cauchy', 'distribution': Distribution.cauchy,
            'density': Density.cauchy},
        {'name': 'Laplace', 'distribution': Distribution.laplace,
            'density': Density.laplace},
        {'name': 'Poisson', 'distribution': Distribution.poisson,
            'density': Density.poisson},
        {'name': 'Uniform', 'distribution': Distribution.uniform,
            'density': Density.uniform}
    ]

    for func in funcs:
        plot(func=func, sizes=size)

    plt.show()


def create_characteristic_tables():
    sizes = [10, 100, 1000]
    distributions = [Distribution.normal, Distribution.uniform,
                     Distribution.poisson, Distribution.laplace, Distribution.cauchy]
    characteristics = [Characteristic.sample_mean, Characteristic.sample_median,
                       Characteristic.z_R, Characteristic.z_Q,  Characteristic.truncated_mean]

    for distribution in distributions:
        latex = f"\\begin{{tabular}}{{|c | c | c | c | c | c|}} \n \hline \multicolumn{{6}}{{|c|}}{{{distribution.__name__}}} \\\\ \n"
        latex += f" \\hline & $\\bar{{x}}$ & $medx$ & $z_R$ & $z_Q$ & $z_{{tr}}$ \n"
        for size in sizes:
            latex += f" \\\\ \\hline $n={size}$ & & & & & \\\\ \n"
            mean = []
            variance = []

            for characteristic in characteristics:
                data = [characteristic(sorted(distribution(size)))
                        for i in range(1000)]
                mean.append(Characteristic.sample_mean(data))
                variance.append(Characteristic.variance(data))

            latex += f" \\hline $E(z)$ \n"
            for x in mean:
                latex += f" &{round(x, 6)}"

            latex += f" \\\\ \n"
            latex += f" \\hline $D(z)$ \n"

            for x in variance:
                latex += f" &{round(x, 6)}"

            latex += f" \\\\ \n"
            latex += f"\\hline $\hat{{E}}(z)$ \n"
            
            for i in range(len(mean)):
                latex += f"&{round(mean[i], Characteristic.correct_digits(np.sqrt(variance[i])))}"

            latex += f" \\\\ \n"
            latex += f"\\hline $E-\sqrt{{D(z)}}$ \n"

            for i in range(len(mean)):
                latex += f"&{round(mean[i] - np.sqrt(variance[i]), 6)}"

            latex += f" \\\\ \n"
            latex += f"\\hline $E+\sqrt{{D(z)}}$ \n"

            for i in range(len(mean)):
                latex += f"&{round(mean[i] + np.sqrt(variance[i]), 6)}"

        latex += f" \\\\ \\hline \n \end{{tabular}}"
        file = open(
            f"Report/CharacteristicTables/{distribution.__name__}Characteristics" + ".tex", "w")
        file.write(latex)


def create_box_plot():
    sizes = [20, 100]
    distributions = [Distribution.normal, Distribution.uniform,
                     Distribution.poisson, Distribution.laplace, Distribution.cauchy]
    for distribution in distributions:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.suptitle(distribution.__name__)
        data = []
        for i in range(len(sizes)):
            data.append(distribution(sizes[i]))

        ax.boxplot(data, vert=False, labels=[f"n={size}" for size in sizes])
        ax.set(xlabel="x", ylabel="n")
        plt.savefig("Report/pictures/BoxPlots/" + distribution.__name__ + "Boxplot.png")

    plt.show()


def proportion_of_emissions():
    distributions = [Distribution.normal, Distribution.cauchy,
                     Distribution.laplace, Distribution.poisson, Distribution.uniform]

    size_list = [20, 100]

    averaging = 1000
    result = []

    latex = f"\\begin{{tabular}}{{| c | c |}} \hline Sample & Share of emissions \\\\ \\hline"
    for distribution in distributions:
        for size in size_list:
            data = []
            for i in range(averaging):
                array = np.array(distribution(size))
                x1 = np.quantile(array, 0.25) - 3 / 2 * \
                    (np.quantile(array, 0.75) - np.quantile(array, 0.25))
                x2 = np.quantile(array, 0.75) + 3 / 2 * \
                    (np.quantile(array, 0.75) - np.quantile(array, 0.25))
                data.append(
                    len(list(filter(lambda x: x < x1 or x > x2, array))) / size)
            result = (sum(data) / len(data)).__round__(2)
            latex += f" {distribution.__name__} n = {size} & {result} \\\\ \\hline \n"
            
    latex += f" \\end{{tabular}}"
    file = open(f"Report/ProporionOfEmissions/" +
                "ProporionOfEmissions.tex", 'w')
    file.write(latex)


def plot_empirical_distribution():
    sizes = [20, 60, 100]
    funcs = [
        {'distribution': [Distribution.normal, Density.normal,
                          CumulativeDensity.normal], 'a': -4, 'b': 4},
        {'distribution': [Distribution.cauchy, Density.cauchy,
                          CumulativeDensity.cauchy], 'a': -4, 'b': 4},
        {'distribution': [Distribution.laplace, Density.laplace,
                          CumulativeDensity.laplace], 'a': -4, 'b': 4},
        {'distribution': [Distribution.poisson, Density.poisson,
                          CumulativeDensity.poisson], 'a': 6, 'b': 14},
        {'distribution': [Distribution.uniform, Density.uniform,
                          CumulativeDensity.uniform], 'a': -4, 'b': 4}
    ]

    for func in funcs:
        fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 3))
        for i in range(len(sizes)):
            a = func['a']
            b = func['b']
            distribution = func['distribution']
            sample = distribution[0](sizes[i])
            ecdf = ECDF(sample)
            x = np.linspace(a, b)
            y_ecdf = ecdf(x)
            y_cdf = distribution[2](x)
            axs[i].step(x, y_ecdf)
            axs[i].plot(x, y_cdf)
            axs[i].set_title(distribution[0].__name__ + ' n=' + str(sizes[i]))
            axs[i].set(xlabel="x", ylabel="F(x)")
            plt.savefig("Report/pictures/empirical_distribution/cdf" + distribution[0].__name__)

    plt.show()


def plot_kde():
    def kde(samples, param):
        n_kde = stats.gaussian_kde(samples, bw_method="silverman")
        n_kde.set_bandwidth(n_kde.factor * param)
        return n_kde

    sizes = [20, 60, 100]
    funcs = [
        {'distribution': [Distribution.normal, Density.normal,
                          CumulativeDensity.normal], 'a': -4, 'b': 4},
        {'distribution': [Distribution.cauchy, Density.cauchy,
                          CumulativeDensity.cauchy], 'a': -4, 'b': 4},
        {'distribution': [Distribution.laplace, Density.laplace,
                          CumulativeDensity.laplace], 'a': -4, 'b': 4},
        {'distribution': [Distribution.poisson, Density.poisson,
                          CumulativeDensity.poisson], 'a': 6, 'b': 14},
        {'distribution': [Distribution.uniform, Density.uniform,
                          CumulativeDensity.uniform], 'a': -4, 'b': 4}
    ]
    bandwidth = [0.5, 1.0, 2.0]

    for func in funcs:
        distribution = func['distribution']
        a = func['a']
        b = func['b']
        for size in sizes:
            sample = distribution[0](size)
            fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 3))

            for bandw, ax in zip(bandwidth, axs):
                ax.set_title(distribution[0].__name__ + ' n=' + str(size))
                ax.set_xlabel(f"h = h_n*{bandwidth}")
                ax.set_ylabel("f(x)")
                
                x = np.linspace(a, b)
                y = distribution[1](x)
                ax.plot(x, y)

                cur_kde = kde(sample, bandw)
                y_kde = cur_kde.evaluate(x)
                ax.plot(x, y_kde)
                ax.set_ylim([0, 1])
                ax.grid()

            plt.savefig("Report/pictures/KDE/kdeN=" + str(size) +
                        " " + distribution[0].__name__)
    # plt.show()


if __name__ == "__main__":
    # plot_distribution_histograms()
    create_characteristic_tables()
    # create_box_plot()
    # proportion_of_emissions()
    # plot_empirical_distribution()
    # plot_kde()
