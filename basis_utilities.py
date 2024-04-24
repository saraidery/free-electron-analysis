import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial2

class Basis: # Currently decontracted, single center even tempered with all l same exponents
    def __init__(self, coefficients, exponents, angular_momenta, center):
        self.coefficients = coefficients
        self.exponents = exponents
        self.angular_momenta = angular_momenta
        self.center = center
        self.n_shells = len(exponents)*len(angular_momenta)

    def n_aos(self):
        n_ao = 0
        for i in np.arange(len(self.exponents)):
            for l in self.angular_momenta:
                for x in range(l, -1, -1):
                    for y in (range(l - x, -1, -1)):
                        n_ao += 1
        return n_ao

def PW_GTO(k, r, A, a, d, x, y, z):
    return GTO(r, A, a, d, x, y, z)*plane_wave(k, r, 1.0)

def GTO(r, A, a, d, x, y, z):
    rA = r - A
    rA2 = np.dot(rA, rA)

    norm = np.sqrt(factorial2(2*(x + y + z) - 1)/(factorial2(2*x-1)*factorial2(2*y-1)*factorial2(2*z-1)))
    angular_part = pow(rA[0],x)*pow(rA[1],y)*pow(rA[2],z)

    radial_part = 0.0
    for i, a_ in enumerate(a):
        radial_part += d[i]*np.exp(-a_*rA2)

    return norm*angular_part*radial_part


def plane_wave(k, r, sign=1):
    kr = np.dot(k, r)
    return complex(np.cos(kr), sign*np.sin(kr))

def evaluate_PWGTOs_at_points(k, r, basis):

    A = basis.center
    value = []
    for k_ in k:
        for i, alpha in enumerate(basis.exponents):
            d = basis.coefficients[i]
            for l in basis.angular_momenta:
                for x in range(l, -1, -1):
                    for y in (range(l - x, -1, -1)):
                        z = l - x - y
                        value.append(PW_GTO(k_, r, A, [alpha], [d], x, y, z))

    return np.array(value)

def evaluate_GTOs_at_points(r, basis):
    A = basis.center
    value = []
    for i, alpha in enumerate(basis.exponents):
        d = basis.coefficients[i]
        for l in basis.angular_momenta:
            for x in range(l, -1, -1):
                for y in (range(l - x, -1, -1)):
                    z = l - x - y
                    value.append(GTO(r, A, [alpha], [d], x, y, z))
    return np.array(value)


def plot_GTO_basis(filename, basis, x_max=10, n_x=100):


    x_ = np.linspace(0, x_max, num=n_x)
    print(x_.shape)

    f = np.zeros([n_x, basis.n_aos()])

    for i, x in enumerate(x_):
        r = np.array([0.0, 0.0, x], dtype=float)
        f[i,:] = evaluate_GTOs_at_points(r, basis)


    fig, ax = plt.subplots()
    for i in np.arange(basis.n_aos()):
        ax.plot(x_, f[:,i], label=f"AO {i}")

    plt.savefig(f"{filename}_GTO.png")
