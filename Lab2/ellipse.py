
class Ellipse:
    def __init__(self, m_expectation_x, m_expectation_y, sigma_x, sigma_y, rho):
        self.a = 1 / (sigma_x * sigma_x)
        self.b = 2 * rho / (sigma_x * sigma_y)
        self.c = 1 / (sigma_y * sigma_y)
        self.x0 = m_expectation_x
        self.y0 = m_expectation_y

    def rad2(self, samples):
        rads = self.z(samples[:, 0], samples[:, 1])
        return max(rads)

    def z(self, x, y):
        values = self.a * ((x - self.x0) ** 2) - \
                 self.b * (x - self.x0) * (y - self.y0) + \
                 self.c * ((y - self.y0) ** 2)
        return values