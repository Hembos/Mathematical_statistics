import numpy as np
from distribution import Distribution
import matplotlib.pyplot as plt


class Model:

    @staticmethod
    def reference(x):
        return 2 + 2 * x

    @staticmethod
    def default(x):
        return 2 + 2 * x + Distribution.normal(20)

    @staticmethod
    def default_with_distrubance(x):
        y = 2 + 2 * x + Distribution.normal(20)
        y[0] = 10
        y[19] = -10
        return y


class Methods:

    @staticmethod
    def least_squares(x, y):
        square_mean = np.mean(list(map(lambda t: t ** 2, x)))
        b_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (square_mean - np.mean(x) ** 2)
        b_0 = np.mean(y) - np.mean(x) * b_1

        return b_0, b_1, b_0 + b_1 * x

    @staticmethod
    def least_absolute_deviations(x, y):
        n = len(x)

        r_Q = np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))

        if n % 4 != 0:
            l_index = n // 4 + 1
        else:
            l_index = n // 4

        j_index = n - l_index + 1

        q_y = (y[j_index] - y[l_index])
        q_x = (x[j_index] - x[l_index])

        b_1 = r_Q * q_y / q_x
        b_0 = np.median(y) - b_1 * np.median(x)

        return b_0, b_1, b_0 + b_1 * x


class Task2:
    def __init__(self):
        self._num_points = 20
        self._start_point = -1.8
        self._end_point = 2.0

    def run(self):
        x = np.linspace(self._start_point, self._end_point, self._num_points, endpoint=True)
        y_ideal = Model.reference(x)

        for func in [Model.default, Model.default_with_distrubance]:
            y = func(x)
            b0_ls, b1_ls, y_ls = Methods.least_squares(x, y)
            b0_lad, b1_lad, y_lad = Methods.least_absolute_deviations(x, y)

            print(f"b0_ls = {b0_ls.__round__(3)} b1_ls = {b1_ls.__round__(3)} b0_lad = {b0_lad.__round__(3)} b1_lad = {b1_lad.__round__(3)}")

            plt.plot(x, y, 'ko', mfc='none')
            plt.plot(x, y_ideal, 'o')
            plt.plot(x, y_ls, 'y')
            plt.plot(x, y_lad, 'r')
            plt.legend(('Выборка', 'Модель', 'МНК', 'МНМ'))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f"Report/images/{func.__name__}")
            plt.show()
