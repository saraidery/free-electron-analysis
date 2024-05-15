import numpy as np

from obara_saika.angular_momentum import get_n_cartesian, get_cartesians, get_n_cartesian_accumulated, get_cartesian_index_accumulated, get_cartesians_accumulated, get_cartesian_index
from obara_saika.GTO import GTO, ShellGTO
from obara_saika.math import boys_kummer

import numpy as np

class ERIGTO:

    def __init__(self, A, alpha, l_a, B, beta, l_b, C, gamma, l_c, D, delta, l_d):

        self.sh_A = ShellGTO(A, alpha, l_a)
        self.sh_B = ShellGTO(B, beta,  l_b)
        self.sh_C = ShellGTO(C, gamma, l_c)
        self.sh_D = ShellGTO(D, delta, l_d)

        [self.p, self.P, self.K_AB] = self.sh_A * self.sh_B
        [self.q, self.Q, self.K_CD] = self.sh_C * self.sh_D

        self.PA = self.P - self.A
        self.PB = self.P - self.B
        self.QC = self.Q - self.C
        self.QD = self.Q - self.D

        self.W = (self.p*self.P + self.q*self.Q)/(self.p + self.q)
        self.WP = self.W - self.P
        self.WQ = self.W - self.Q
        self.PQ = self.P - self.Q

    def normalization_array(self):
        #
        #   Set up matrix of normalization factors for
        #   shell pair
        #
        normalization = np.zeros([get_n_cartesian(self.l_a), get_n_cartesian(self.l_b), get_n_cartesian(self.l_c), get_n_cartesian(self.l_d)])

        cart_a = get_cartesians(self.l_a)
        cart_b = get_cartesians(self.l_b)
        cart_c = get_cartesians(self.l_c)
        cart_d = get_cartesians(self.l_d)

        for i, c_a in enumerate(cart_a):
            for j, c_b in enumerate(cart_b):
                 for k, c_c in enumerate(cart_c):
                     for l, c_d in enumerate(cart_d):

                         gto_A = GTO(self.alpha, self.A, c_a)
                         gto_B = GTO(self.beta, self.B, c_b)
                         gto_C = GTO(self.gamma, self.C, c_c)
                         gto_D = GTO(self.delta, self.D, c_d)
                         normalization[i, j, k, l] =( gto_A.normalization_3d()
                                                     * gto_B.normalization_3d()
                                                     * gto_C.normalization_3d()
                                                     * gto_D.normalization_3d())

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

    @property
    def C(self):
        return self.sh_C.A

    @property
    def D(self):
        return self.sh_D.A

    @property
    def l_c(self):
        return self.sh_C.l

    @property
    def l_d(self):
        return self.sh_D.l

    @property
    def gamma(self):
        return self.sh_C.alpha

    @property
    def delta(self):
        return self.sh_D.alpha

    def do_recurrence_bra(self, a, b, c, d, cart, PX, WP, aux, m):

        rho = self.p*self.q/(self.p + self.q)
        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]
        c_q = c[idx_cart]
        d_q = d[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)
        c_c = get_cartesian_index_accumulated(c)
        c_d = get_cartesian_index_accumulated(d)

        value = (
                 + PX[idx_cart] * aux[c_a, c_b, c_c, c_d, m]
                 + WP[idx_cart] * aux[c_a, c_b, c_c, c_d, m + 1])

        if (a_q > 0):
            c_a_m = get_cartesian_index_accumulated(a-cart)
            value += 1.0/(2.0 * self.p)*a_q*(aux[c_a_m, c_b, c_c, c_d, m] - (rho/self.p)*aux[c_a_m, c_b, c_c, c_d, m + 1])
        if (b_q > 0):
            c_b_m = get_cartesian_index_accumulated(b-cart)
            value += 1.0/(2.0 * self.p)*b_q*(aux[c_a, c_b_m, c_c, c_d, m] - (rho/self.p)*aux[c_a, c_b_m, c_c, c_d, m + 1])
        if (c_q > 0):
            c_c_m = get_cartesian_index_accumulated(c-cart)
            value += 1.0/(2.0*(self.p+self.q))*c_q*aux[c_a, c_b, c_c_m, c_d, m + 1]
        if (d_q > 0):
            c_d_m = get_cartesian_index_accumulated(d-cart)
            value += 1.0/(2.0*(self.p+self.q))*d_q*aux[c_a, c_b, c_c, c_d_m, m + 1]

        return value

    def do_recurrence_ket(self, a, b, c, d, cart, PX, WP, aux, m):

        rho = self.p*self.q/(self.p + self.q)
        idx_cart = np.argmax(cart)
        a_q = a[idx_cart]
        b_q = b[idx_cart]
        c_q = c[idx_cart]
        d_q = d[idx_cart]

        c_a = get_cartesian_index_accumulated(a)
        c_b = get_cartesian_index_accumulated(b)
        c_c = get_cartesian_index_accumulated(c)
        c_d = get_cartesian_index_accumulated(d)

        value = (
                 + PX[idx_cart] * aux[c_a, c_b, c_c, c_d, m]
                 + WP[idx_cart] * aux[c_a, c_b, c_c, c_d, m + 1])

        if (c_q > 0):
            c_c_m = get_cartesian_index_accumulated(c-cart)
            value += 1.0/(2.0 * self.q)*c_q*(aux[c_a, c_b, c_c_m, c_d, m] - (rho/self.q)*aux[c_a, c_b, c_c_m, c_d, m + 1])
        if (d_q > 0):
            c_d_m = get_cartesian_index_accumulated(d-cart)
            value += 1.0/(2.0 * self.q)*d_q*(aux[c_a, c_b, c_c, c_d_m, m] - (rho/self.q)*aux[c_a, c_b, c_c, c_d_m, m + 1])
        if (a_q > 0):
            c_a_m = get_cartesian_index_accumulated(a-cart)
            value += 1.0/(2.0*(self.p+self.q))*a_q*aux[c_a_m, c_b, c_c, c_d, m + 1]
        if (b_q > 0):
            c_b_m = get_cartesian_index_accumulated(b-cart)
            value += 1.0/(2.0*(self.p+self.q))*b_q*aux[c_a, c_b_m, c_c, c_d, m + 1]

        return value

    def auxiliary_integral_s(self, m):

        rho = self.p*self.q/(self.p + self.q)
        gto_s_P = GTO(self.p, np.real(self.P), np.array([0, 0, 0], dtype=int))
        gto_s_Q = GTO(self.q, np.real(self.Q), np.array([0, 0, 0], dtype=int))

        T = rho * np.dot(self.PQ, self.PQ)

        return 2.0*pow(rho/np.pi,0.5)*self.K_AB*self.K_CD*boys_kummer(m, T)*gto_s_P.GTO_s_overlap_3d()*gto_s_Q.GTO_s_overlap_3d()


    def integral_accumulated(self, aux, PA, PB, QC, QD):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)
        dim_c = get_n_cartesian_accumulated(self.l_c)
        dim_d = get_n_cartesian_accumulated(self.l_d)

        dim_m = self.l_a + self.l_b + self.l_c + self.l_d + 1 # HOW MANY DO WE NEED???

        for i in np.arange(dim_m):
            aux[0, 0, 0, 0, i] = self.auxiliary_integral_s(i)

        incr = [np.array([0, 0, 0], dtype=int),
                np.array([1, 0, 0], dtype=int),
                np.array([0, 1, 0], dtype=int),
                np.array([0, 0, 1], dtype=int)]


        for a in get_cartesians_accumulated(self.l_a):
            for b in get_cartesians_accumulated(self.l_b):
                for c in get_cartesians_accumulated(self.l_c):
                    for d in get_cartesians_accumulated(self.l_d):
                        for i in incr:
                            if (sum(a + i) > self.l_a):
                                continue
                            for j in incr:
                                if (sum(b + j) > self.l_b):
                                    continue
                                for k in incr:
                                    if (sum(c + k) > self.l_c):
                                        continue
                                    for l in incr:
                                        if (sum(d + l) > self.l_d):
                                            continue
                                        if (sum(i) + sum(j) + sum(k) + sum(l) == 0):
                                            continue

                                        c_a = get_cartesian_index_accumulated(a + i)
                                        c_b = get_cartesian_index_accumulated(b + j)
                                        c_c = get_cartesian_index_accumulated(c + k)
                                        c_d = get_cartesian_index_accumulated(d + l)

                                        for m in np.arange(self.l_b + self.l_a + self.l_c + self.l_d - sum(a) - sum(b) - sum(c) - sum(d)):
                                            if (sum(l) > 0):
                                                aux[c_a, c_b, c_c, c_d, m] = self.do_recurrence_ket(a + i, b + j, c + k, d, l, self.QD, self.WQ, aux, m)
                                            elif (sum(k) > 0):
                                                aux[c_a, c_b, c_c, c_d, m] = self.do_recurrence_ket(a + i, b + j, c, d + l, k, self.QC, self.WQ, aux, m)
                                            elif (sum(j) > 0):
                                                aux[c_a, c_b, c_c, c_d, m] = self.do_recurrence_bra(a + i, b, c + k, d + l, j, self.PB, self.WP, aux, m)
                                            elif (sum(i) > 0):
                                                aux[c_a, c_b, c_c, c_d, m] = self.do_recurrence_bra(a, b + j, c + k, d + l, i, self.PA, self.WP, aux, m)


    def integral(self):

        dim_a = get_n_cartesian_accumulated(self.l_a)
        dim_b = get_n_cartesian_accumulated(self.l_b)
        dim_c = get_n_cartesian_accumulated(self.l_c)
        dim_d = get_n_cartesian_accumulated(self.l_d)

        dim_m = self.l_a + self.l_b + self.l_c + self.l_d + 1
        aux = np.zeros([dim_a, dim_b, dim_c, dim_d, dim_m])

        self.integral_accumulated(aux, self.PA, self.PB, self.QC, self.QD)

        extract_a = dim_a - get_n_cartesian(self.l_a)
        extract_b = dim_b - get_n_cartesian(self.l_b)
        extract_c = dim_c - get_n_cartesian(self.l_c)
        extract_d = dim_d - get_n_cartesian(self.l_d)

        g = aux[extract_a:,extract_b:, extract_c:, extract_d:, 0]

        normalization = self.normalization_array()

        return np.multiply(g, normalization)
