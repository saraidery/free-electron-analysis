import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial2
from even_tempered_basis import *

class Shell:
    def __init__(self, A, exponents, coefficients, l):
        self.A = A
        self.exponents = exponents
        self.coefficients = coefficients
        self.l = l

    def __repr__(self):
        return f"exp: {self.exponents}, coef: {self.coefficients}, cen: {self.A}, ang: {self.l}"


class Basis:
    def __init__(self, shell_list):
        self.shells = shell_list

    def n_aos(self):
        n_ao = 0
        for shell in self.shells:
            l = shell.l
            for x in range(l, -1, -1):
                for y in (range(l - x, -1, -1)):
                    n_ao += 1
        return n_ao


class EvenTemperedBasis(Basis):
    def __init__(self, N, l, beta, initial_alpha, centers, normalize):

        self.shells = []

        for center in centers:
            for i, l_ in enumerate(l):
                n = N[i]
                exponents = generate_even_tempered_from_least_diffuse(initial_alpha, n, beta)
                for alpha in exponents:
                    d = 1.0
                    if normalize:
                        d = 1.0/(pow(l_/(2.0*alpha), l_/2)*np.exp(-l_/2.0))
                    s = Shell(center, [alpha], [d], l_)
                    self.shells.append(s)


class QchemBasis(Basis):
    def __init__(self, filename):

        self.shells = []
        n_shells, exponents, angular_momentum, coefficients, center_index, centers = self.read_qchem_basis(filename)
        for i in np.arange(n_shells):
            s = Shell(centers[center_index[i]], exponents[i], coefficients[i], angular_momentum[i])
            self.shells.append(s)

    def read_qchem_basis(self, filename):

        file = open(filename, 'r')
        lines = file.readlines()

        lines = [i for i in lines if i != "\n"]
        lines = [i.strip("\n") for i in lines]
        lines = [i.lstrip() for i in lines]

        offset = 0

        # first line
        current_line = lines[offset]
        offset += 1

        n_centers = int(current_line.split(" ")[0])
        n_exponents = int(current_line.split(" ")[1])
        n_coefficients = int(current_line.split(" ")[2])
        n_shells = int(current_line.split(" ")[3])

        centers      = np.zeros(n_centers);
        exponents    = np.zeros(n_exponents);
        coefficients = np.zeros(n_coefficients);
        shells       = np.zeros(n_shells);

        centers = np.zeros([n_centers, 3])
        for i in np.arange(n_centers):

            current_line = lines[i + offset].strip()
            coordinates = [i for i in current_line.split(" ") if i != ""]

            centers[i, 0] = float(coordinates[0])
            centers[i, 1] = float(coordinates[1])
            centers[i, 2] = float(coordinates[2])

        offset += n_centers

        for i in np.arange(n_exponents):
            current_line = lines[offset + i].strip()
            exponents[i] = float(current_line)

        offset += n_exponents

        for i in np.arange(n_coefficients):
            current_line = lines[offset + i].strip()
            coefficients[i] = float(current_line)

        offset += n_coefficients

        exponent_offset = np.zeros(n_shells, dtype=int)
        shared_exponents = np.zeros(n_shells, dtype=int)
        degree_of_contraction = np.zeros(n_shells, dtype=int)

        coefficient_offset = []
        angular_momentum = []
        center_index = []

        for i in np.arange(n_shells):
            current_line = lines[offset].strip()
            line_1 = [i for i in current_line.split(" ") if i != ""]

            exponent_offset[i] = int(line_1[3])
            shared_exponents[i] = int(line_1[4])

            current_line = lines[offset + 1].strip()
            offsets = [i for i in current_line.split(" ") if i != ""]

            for j in np.arange(shared_exponents[i]):
                center_index.append(int(line_1[0]))
                coefficient_offset.append(int(offsets[j*2]))
                angular_momentum.append(int(offsets[j*2+1]))

            current_line = lines[offset + 2].strip()
            offsets = [i for i in current_line.split(" ") if i != ""]

            degree_of_contraction[i] = int(offsets[0])

            offset += 3

        exponents_expanded = []
        coefficients_expanded = []

        count_shells = 0
        for i in np.arange(n_shells):
            for j in np.arange(shared_exponents[i]):
                exponents_expanded.append(exponents[exponent_offset[i]:exponent_offset[i]+degree_of_contraction[i]])
                coefficients_expanded.append(coefficients[coefficient_offset[count_shells]:coefficient_offset[count_shells]+degree_of_contraction[i]])

                count_shells += 1

        n_shells = count_shells

        return n_shells, exponents_expanded, angular_momentum, coefficients_expanded, center_index, centers

def PW_GTO(k, r, A, a, d, x, y, z, k_scale_angular_momentum):

    result = GTO(r, A, a, d, x, y, z)*plane_wave(k, r, 1.0)
    if k_scale_angular_momentum:
        kA = k - A
        result *= pow(kA[0],x)*pow(kA[1],y)*pow(kA[2],z)

    return result

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

def evaluate_PWGTOs_at_points(k, r, basis, k_scale_angular_momentum=False):
    value = []
    for k_ in k:
        for s in basis.shells:
            l = s.l
            for x in range(l, -1, -1):
                for y in (range(l - x, -1, -1)):
                    z = l - x - y
                    value.append(PW_GTO(k_, r, s.A, s.exponents, s.coefficients, x, y, z, k_scale_angular_momentum))

    return np.array(value)

def evaluate_GTOs_at_points(r, basis):

    value = []
    for s in basis.shells:
        l = s.l
        for x in range(l, -1, -1):
            for y in (range(l - x, -1, -1)):
                z = l - x - y
                value.append(GTO(r, s.A, s.exponents, s.coefficients, x, y, z))

    return np.array(value)


def plot_GTO_basis(filename, basis, x_max=30, n_x=100):

    x_ = np.linspace(0, x_max, num=n_x)
    f = np.zeros([n_x, basis.n_aos()])

    for i, x in enumerate(x_):
        r = np.array([0.0, 0.0, x], dtype=float)
        f[i,:] = evaluate_GTOs_at_points(r, basis)


    fig, ax = plt.subplots()
    for i in np.arange(basis.n_aos()):
        ax.plot(x_, f[:,i], label=f"AO {i}")

    plt.savefig(f"{filename}_GTO.png")
