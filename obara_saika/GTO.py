import mpmath as mp
import numpy as np
from scipy.special import factorial2

class GTO:

    def __init__(self, alpha, A, a):
        self.A = A
        self.a = a
        self.alpha = alpha

    def GTO_s_overlap_3d(self):
        I_3d = (
            self.GTO_s_overlap_1d()
            * self.GTO_s_overlap_1d()
            * self.GTO_s_overlap_1d()
        )
        return I_3d

    def GTO_s_overlap_1d(self):
        # Integrates exp(-alpha (x)^2) from -infinity to infinity
        #
        return np.sqrt(np.pi / self.alpha)


    def normalization_3d(self):
        # Non-axial normalization used in QChem for l > 2
        #
        if (sum(self.a) < 2):
            return 1.0

        numerator = (
            factorial2(2 * self.a[0] - 1) * factorial2(2 * self.a[1] - 1) * factorial2(2 * self.a[2] - 1)
        )

        denominator = factorial2(2*sum(self.a) - 1)
        non_axial_normalization = pow(denominator / numerator, 0.5)

        return non_axial_normalization


class PWGTO(GTO):

    def __init__(self, alpha, A, a, k):
        self.A = A
        self.a = a
        self.alpha = alpha
        self.k = k

class ShellGTO:

    def __init__(self, A, alpha, l):
        self.A = A
        self.alpha = alpha
        self.l = l

    def __repr__(self):
        return f"exp: {self.alpha},  cen: {self.A}, ang: {self.l}"


    def __mul__(self, other):
        #
        #   Compute new center, exponent and pre-exponential factor for
        #   product of two GTOs
        #
        p = self.alpha + other.alpha
        P = (self.alpha * self.A + other.alpha * other.A)/p
        AB = self.A - other.A
        K = np.exp(-(self.alpha * other.alpha  / p) * np.dot(AB, AB))

        return p, P, K

class ShellPWGTO:

    def __init__(self, A, alpha, l, k):
        self.A = A
        self.alpha = alpha
        self.l = l
        self.k = k

    def __repr__(self):
        return f"exp: {self.alpha},  cen: {self.A}, ang: {self.l}, mom: {self.k}"


    def __mul__(self, other):
        #
        #   Compute new center, exponent, pre-exponential factor, and complex centers for
        #   product of two PWGTOs
        #
        p = self.alpha + other.alpha
        Ad = self.A + 1j * self.k/(2.0*self.alpha)
        Bd = other.A + 1j * other.k/(2.0*other.alpha)

        P = (self.alpha * Ad + other.alpha * Bd)/p

        AB = Ad - Bd

        K = (np.exp(- (self.alpha * other.alpha  / p) * np.dot(AB, AB))
            * np.exp(- np.dot(self.k, self.k) / (4.0 * self.alpha))
            * np.exp(- np.dot(other.k,other.k) / (4.0 * other.alpha)))

        return p, P, K, Ad, Bd

