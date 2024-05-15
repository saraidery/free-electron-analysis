import numpy as np

from obara_saika.angular_momentum import get_n_cartesian, get_cartesians, get_n_cartesian_accumulated, get_cartesian_index_accumulated, get_cartesians_accumulated
from obara_saika.GTO import GTO, ShellGTO
from obara_saika.math import boys_kummer

import numpy as np

class BaseIntegralGTO:

    def normalization_array(self):
        #
        #   Set up matrix of normalization factors for
        #   shell pair
        #
        normalization = np.zeros([get_n_cartesian(self.l_a), get_n_cartesian(self.l_b)])

        cart_a = get_cartesians(self.l_a)
        cart_b = get_cartesians(self.l_b)

        for i, c_a in enumerate(cart_a):
            for j, c_b in enumerate(cart_b):

                gto_A = GTO(self.alpha, self.A, c_a)
                gto_B = GTO(self.beta, self.B, c_b)
                normalization[i, j] =(gto_A.normalization_3d()
                                        * gto_B.normalization_3d())

        return normalization

    @property
    def A(self):
        return self.sh_A.A

    @property
    def B(self):
        return self.sh_B.A

    @property
    def l_a(self):
        return self.sh_A.l

    @property
    def l_b(self):
        return self.sh_B.l

    @property
    def alpha(self):
        return self.sh_A.alpha

    @property
    def beta(self):
        return self.sh_B.alpha

class OverlapIntegralGTO(BaseIntegralGTO):

    def __init__(self, A, alpha, l_a, B, beta, l_b):

        self.sh_A = ShellGTO(A, alpha, l_a)
        self.sh_B = ShellGTO(B, beta, l_b)

        [self.p, self.P, self.K] = self.sh_A * self.sh_B

        self.PA = self.P - self.A
        self.PB = self.P - self.B

    def do_recurrence(self, a, b, cart, PX, I):

        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)

        value = PX[idx_cart] * I[c_a, c_b]

        if (a_q > 0):
           c_a_m = get_cartesian_index_accumulated(a-cart)
           value += 1.0/(2.0 * self.p)*a_q*(I[c_a_m, c_b])
        if (b_q > 0):
           c_b_m = get_cartesian_index_accumulated(b-cart)
           value += 1.0/(2.0 * self.p)*b_q*(I[c_a, c_b_m])

        return value

    def integral_accumulated(self, I, PA, PB):

        gto_s_P = GTO(self.p, np.real(self.P), np.array([0, 0, 0], dtype=int))

        I[0, 0] = self.K*gto_s_P.GTO_s_overlap_3d()

        incr = [np.array([0, 0, 0], dtype=int),
                np.array([1, 0, 0], dtype=int),
                np.array([0, 1, 0], dtype=int),
                np.array([0, 0, 1], dtype=int)]

        if (self.l_a == 0):
            for b in get_cartesians_accumulated(self.l_b-1):
                for j in incr:
                    if (sum(j) == 0):
                        continue

                    a = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a)
                    c_b = get_cartesian_index_accumulated(b + j)

                    I[c_a, c_b] = self.do_recurrence(a, b, j, PB, I)

        if (self.l_b == 0):
            for a in get_cartesians_accumulated(self.l_a-1):
                for i in incr:
                    if (sum(i) == 0):
                        continue

                    b = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a + i)
                    c_b = get_cartesian_index_accumulated(b)

                    I[c_a, c_b] = self.do_recurrence(a, b, i, PA, I)

        for a in get_cartesians_accumulated(self.l_a-1):
            for b in get_cartesians_accumulated(self.l_b-1):

                for i in incr:
                    for j in incr:
                        if (np.sum(i) + np.sum(j) == 0):
                            continue

                        c_a = get_cartesian_index_accumulated(a + i)
                        c_b = get_cartesian_index_accumulated(b + j)

                        if (np.sum(i) == 0):
                            I[c_a, c_b] = self.do_recurrence(a, b, j, PB, I)
                        if (np.sum(j) == 0):
                            I[c_a, c_b] = self.do_recurrence(a, b, i, PA, I)
                        if (np.sum(i) == np.sum(j)):
                            I[c_a, c_b] = self.do_recurrence(a, b+j, i, PA, I)

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        I = np.zeros([dim_a, dim_b])

        self.integral_accumulated(I, self.PA, self.PB)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        S_shp = I[extract_a:,extract_b:]

        normalization = self.normalization_array()

        return np.multiply(S_shp, normalization)


class KineticIntegralGTO(BaseIntegralGTO):

    def __init__(self, A, alpha, l_a, B, beta, l_b):

        self.sh_A = ShellGTO(A, alpha, l_a)
        self.sh_B = ShellGTO(B, beta, l_b)

        [self.p, self.P, self.K] = self.sh_A * self.sh_B

        self.PA = self.P - self.A
        self.PB = self.P - self.B


    def do_recurrence(self, a, b, cart, PX, I):

        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)

        value = PX[idx_cart] * I[c_a, c_b]

        if (a_q > 0):
           c_a_m = get_cartesian_index_accumulated(a-cart)
           value += 1.0/(2.0 * self.p)*a_q*(I[c_a_m, c_b])
        if (b_q > 0):
           c_b_m = get_cartesian_index_accumulated(b-cart)
           value += 1.0/(2.0 * self.p)*b_q*(I[c_a, c_b_m])

        return value

    def overlap_recurrence(self, a, cart, S, b, beta):

        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]

        c_a_p = get_cartesian_index_accumulated(a+cart)
        c_b = get_cartesian_index_accumulated(b)

        value = 2.0*self.alpha*self.beta/self.p*S[c_a_p, c_b]

        if (a_q > 0):
           c_a_m = get_cartesian_index_accumulated(a-cart)
           value -= beta/self.p *a_q * S[c_a_m, c_b]

        return value

    def get_integral_over_s(self, S_00):

        AB = self.A - self.B
        return  self.alpha*self.beta/self.p*(3.0 - 2.0 * (self.alpha*self.beta/self.p) * np.dot(AB,AB))*S_00

    def integral_accumulated(self, I, PA, PB, S):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        gto_s_P = GTO(self.p, np.real(self.P), np.array([0, 0, 0], dtype=int))

        I[0, 0] = self.get_integral_over_s(S[0,0])

        incr = [np.array([0, 0, 0], dtype=int),
                np.array([1, 0, 0], dtype=int),
                np.array([0, 1, 0], dtype=int),
                np.array([0, 0, 1], dtype=int)]

        if (self.l_a == 0):
            for b in get_cartesians_accumulated(self.l_b-1):
                for j in incr:
                    if (sum(j) == 0):
                        continue

                    a = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a)
                    c_b = get_cartesian_index_accumulated(b + j)

                    I[c_a, c_b] = self.do_recurrence(a, b, j, PB, I)
                    I[c_a, c_b] += self.overlap_recurrence(b, j, S.T, a, self.alpha)

        if (self.l_b == 0):
            for a in get_cartesians_accumulated(self.l_a-1):
                for i in incr:
                    if (sum(i) == 0):
                        continue

                    b = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a + i)
                    c_b = get_cartesian_index_accumulated(b)

                    I[c_a, c_b] = self.do_recurrence(a, b, i, PA, I)
                    I[c_a, c_b] += self.overlap_recurrence(a, i, S, b, self.beta)

        for a in get_cartesians_accumulated(self.l_a-1):
            for b in get_cartesians_accumulated(self.l_b-1):

                for i in incr:
                    for j in incr:
                        if (np.sum(i) + np.sum(j) == 0):
                            continue

                        c_a = get_cartesian_index_accumulated(a + i)
                        c_b = get_cartesian_index_accumulated(b + j)

                        if (np.sum(i) == 0):
                            I[c_a, c_b] = self.do_recurrence(a, b, j, PB, I)
                            I[c_a, c_b] += self.overlap_recurrence(b, j, S.T, a, self.alpha)
                        if (np.sum(j) == 0):
                            I[c_a, c_b] = self.do_recurrence(a, b, i, PA, I)
                            I[c_a, c_b] += self.overlap_recurrence(a, i, S, b, self.beta)
                        if (np.sum(i) == np.sum(j)):
                            I[c_a, c_b] = self.do_recurrence(a, b+j, i, PA, I)
                            I[c_a, c_b] += self.overlap_recurrence(a, i, S, b+j, self.beta)


    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        overlap = OverlapIntegralGTO(self.A, self.alpha, self.l_a + 1, self.B, self.beta, self.l_b + 1)

        S = np.zeros([get_n_cartesian_accumulated(self.l_a+1), get_n_cartesian_accumulated(self.l_b+1)])
        overlap.integral_accumulated(S, self.PA, self.PB)

        I = np.zeros([dim_a, dim_b])

        self.integral_accumulated(I, self.PA, self.PB, S)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        S_shp = I[extract_a:,extract_b:]

        normalization = self.normalization_array()

        return np.multiply(S_shp, normalization)

class NucAttIntegralGTO(BaseIntegralGTO):

    def __init__(self, A, alpha, l_a, B, beta, l_b, C, Z):

        self.sh_A = ShellGTO(A, alpha, l_a)
        self.sh_B = ShellGTO(B, beta, l_b)
        self.C = C
        self.Z = Z

        [self.p, self.P, self.K] = self.sh_A * self.sh_B

        self.PA = self.P - self.A
        self.PB = self.P - self.B
        self.PC = self.P - self.C


    def do_recurrence(self, a, b, cart, PX, aux, m):

        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)

        value = (
                 + PX[idx_cart] * aux[c_a, c_b, m]
                 - self.PC[idx_cart] * aux[c_a, c_b, m + 1])

        if (a_q > 0):
           c_a_m = get_cartesian_index_accumulated(a-cart)
           value += 1.0/(2.0 * self.p)*a_q*(aux[c_a_m, c_b, m] - aux[c_a_m, c_b, m + 1])
        if (b_q > 0):
           c_b_m = get_cartesian_index_accumulated(b-cart)
           value += 1.0/(2.0 * self.p)*b_q*(aux[c_a, c_b_m, m] - aux[c_a, c_b_m, m + 1])

        return value

    def auxiliary_integral_s(self, m):

        U = self.p * np.dot(self.PC, self.PC)

        gto_s_P = GTO(self.p, np.real(self.P), np.array([0, 0, 0], dtype=int))

        return self.K * 2.0 * pow(self.p / np.pi, 0.5) * gto_s_P.GTO_s_overlap_3d() * boys_kummer(m, U)


    def integral_accumulated(self, aux, PA, PB):


        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        for i in np.arange(dim_a + dim_b + 1):
            aux[0, 0, i] = self.auxiliary_integral_s(i)

        incr = [np.array([0, 0, 0], dtype=int),
                np.array([1, 0, 0], dtype=int),
                np.array([0, 1, 0], dtype=int),
                np.array([0, 0, 1], dtype=int)]

        if (self.l_a == 0):
            for b in get_cartesians_accumulated(self.l_b-1):
                for j in incr:
                    if (sum(j) == 0):
                        continue

                    a = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a)
                    c_b = get_cartesian_index_accumulated(b + j)

                    for m in np.arange(self.l_b + self.l_a + 1 - sum(a) - sum(b)):
                        aux[c_a, c_b, m] = self.do_recurrence(a, b, j, PB, aux, m)

        if (self.l_b == 0):
            for a in get_cartesians_accumulated(self.l_a-1):
                for i in incr:
                    if (sum(i) == 0):
                        continue

                    b = np.array([0, 0, 0], dtype=int)
                    c_a = get_cartesian_index_accumulated(a + i)
                    c_b = get_cartesian_index_accumulated(b)

                    for m in np.arange(self.l_b + self.l_a + 1 - sum(a) - sum(b)):
                        aux[c_a, c_b, m] = self.do_recurrence(a, b, i, PA, aux, m)

        for a in get_cartesians_accumulated(self.l_a-1):
            for b in get_cartesians_accumulated(self.l_b-1):

                for i in incr:
                    for j in incr:
                        if (np.sum(i) + np.sum(j) == 0):
                            continue

                        c_a = get_cartesian_index_accumulated(a + i)
                        c_b = get_cartesian_index_accumulated(b + j)

                        for m in np.arange(self.l_b + self.l_a + 1 - sum(a) - sum(b)):
                            if (np.sum(i) == 0):
                                aux[c_a, c_b, m] = self.do_recurrence(a, b, j, PB, aux, m)
                            if (np.sum(j) == 0):
                                aux[c_a, c_b, m] = self.do_recurrence(a, b, i, PA, aux, m)
                            if (np.sum(i) == np.sum(j)):
                                aux[c_a, c_b, m] = self.do_recurrence(a, b+j, i, PA, aux, m)

    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)

        aux = np.zeros([dim_a, dim_b, dim_a + dim_b + 1])

        self.integral_accumulated(aux, self.PA, self.PB)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)

        V_shp = -self.Z*aux[extract_a:,extract_b:, 0]

        normalization = self.normalization_array()

        return np.multiply(V_shp, normalization)