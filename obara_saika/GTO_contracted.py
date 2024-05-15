import numpy as np

from obara_saika.angular_momentum import get_n_cartesian
from obara_saika.GTO_integrals import OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO


import numpy as np


class OverlapGTO:

    def __init__(self, basis):
        self.basis = basis

    def get_overlap(self):

        S = np.zeros([self.basis.sh_dim, self.basis.sh_dim])

        for sh_A in self.basis:
            for sh_B in self.basis:

                I = ContractedOverlapIntegralGTO(sh_A.exp, sh_A.coeff, sh_A.center, sh_A.l,
                                                 sh_B.exp, sh_B.coeff, sh_B.center, sh_B.l)

                S[sh_A.start:sh_A.stop, sh_B.start:sh_B.stop] = I.get_integral()

        return S


class KineticGTO:

    def __init__(self, basis):
        self.basis = basis

    def get_kinetic(self):

        T = np.zeros([self.basis.sh_dim, self.basis.sh_dim])

        for sh_A in self.basis:
            for sh_B in self.basis:

                I = ContractedKineticIntegralGTO(sh_A.exp, sh_A.coeff, sh_A.center, sh_A.l,
                                                 sh_B.exp, sh_B.coeff, sh_B.center, sh_B.l)

                T[sh_A.start:sh_A.stop, sh_B.start:sh_B.stop] = I.get_integral()

        return T

class NucAttGTO:

    def __init__(self, basis, pc):
        self.basis = basis
        self.pc = pc

    def get_nuclear_attraction(self):

        V = np.zeros([self.basis.sh_dim, self.basis.sh_dim])

        for sh_A in self.basis:
            for sh_B in self.basis:

                for point_charge in self.pc:

                    I = ContractedNucAttIntegralGTO(sh_A.exp, sh_A.coeff, sh_A.center, sh_A.l,
                                                    sh_B.exp, sh_B.coeff, sh_B.center, sh_B.l,
                                                    point_charge.center, point_charge.charge)
                    V[sh_A.start:sh_A.stop, sh_B.start:sh_B.stop] += I.get_integral()

        return V

class ContractedOverlapIntegralGTO:

    def __init__(self, exp_A, coeff_A, A, l_a, exp_B, coeff_B, B, l_b):

        self.exp_A = exp_A
        self.coeff_A = coeff_A
        self.A = A
        self.l_a = l_a

        self.exp_B = exp_B
        self.coeff_B = coeff_B
        self.B = B
        self.l_b = l_b


    def get_integral(self):

        dim_a = get_n_cartesian(self.l_a)
        dim_b = get_n_cartesian(self.l_b)

        I = np.zeros([dim_a, dim_b])

        for alpha, c_A in zip(self.exp_A, self.coeff_A):
            for beta, c_B in zip(self.exp_B, self.coeff_B):

                tmp = OverlapIntegralGTO(self.A, alpha, self.l_a, self.B, beta, self.l_b)

                I += c_A * c_B * tmp.integral()

        return I


class ContractedKineticIntegralGTO:

    def __init__(self, exp_A, coeff_A, A, l_a, exp_B, coeff_B, B, l_b):

        self.exp_A = exp_A
        self.coeff_A = coeff_A
        self.A = A
        self.l_a = l_a

        self.exp_B = exp_B
        self.coeff_B = coeff_B
        self.B = B
        self.l_b = l_b


    def get_integral(self):

        dim_a = get_n_cartesian(self.l_a)
        dim_b = get_n_cartesian(self.l_b)

        I = np.zeros([dim_a, dim_b])

        for alpha, c_A in zip(self.exp_A, self.coeff_A):
            for beta, c_B in zip(self.exp_B, self.coeff_B):

                tmp = KineticIntegralGTO(self.A, alpha, self.l_a, self.B, beta, self.l_b)
                I += c_A * c_B * tmp.integral()

        return I

class ContractedNucAttIntegralGTO:

    def __init__(self, exp_A, coeff_A, A, l_a, exp_B, coeff_B, B, l_b, C, Z):

        self.exp_A = exp_A
        self.coeff_A = coeff_A
        self.A = A
        self.l_a = l_a

        self.exp_B = exp_B
        self.coeff_B = coeff_B
        self.B = B
        self.l_b = l_b

        self.C = C
        self.Z = Z


    def get_integral(self):

        dim_a = get_n_cartesian(self.l_a)
        dim_b = get_n_cartesian(self.l_b)

        I = np.zeros([dim_a, dim_b])

        for alpha, c_A in zip(self.exp_A, self.coeff_A):
            for beta, c_B in zip(self.exp_B, self.coeff_B):

                tmp = NucAttIntegralGTO(self.A, alpha, self.l_a, self.B, beta, self.l_b, self.C, self.Z)
                I += c_A * c_B * tmp.integral()

        return I