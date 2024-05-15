import numpy as np
import pytest
import os

from obara_saika import OverlapIntegralGTO, NucAttIntegralGTO, KineticIntegralGTO, QchemBasis, OverlapGTO, KineticGTO, NucAttGTO, PointCharge

class TestGTO:
    def __overlap__(self, l_a, l_b, S_ref):

        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])

        alpha = 0.8
        beta = 1.1

        S = OverlapIntegralGTO(A, alpha, l_a, B, beta, l_b)

        assert np.allclose(S.integral().flatten(), S_ref.flatten())

    def test_overlap_ss(self):
        S_ref = np.array([0.68913535])
        self.__overlap__(0, 0, S_ref)

    def test_overlap_ps(self):
        S_ref = np.array([0.11969193,  -0.47876772, -0.37902444])
        self.__overlap__(1, 0, S_ref)


    def test_overlap_pp(self):
        S_ref = np.array([[ 0.16623243,  0.06047592,  0.04787677],
                 [ 0.06047592, -0.06055228, -0.19150709],
                 [ 0.04787677, -0.19150709,  0.02974163]])
        self.__overlap__(1, 1, S_ref)


    def test_overlap_sd(self):
        S_ref = np.array([0.19234703, -0.07617991, -0.06030909,  0.35728136,  0.24123637,  0.29161306])
        self.__overlap__(0, 2, S_ref)


    def test_overlap_dp(self):
        S_ref = np.array([[ 0.03746228,  0.1021339,   0.080856  ],
                 [-0.20003072, -0.01821594, -0.05761105],
                 [-0.15835766, -0.05761105,  0.00894717],
                 [-0.0649224 ,  0.00770658,  0.20558759],
                 [-0.05761105,  0.05768379, -0.03578868],
                 [-0.04923977,  0.19695908, -0.04356061]])
        self.__overlap__(2, 1, S_ref)


    def test_overlap_dd(self):
        S_ref = np.array([[ 0.13595341,  0.03278479,  0.02595463,  0.10479923,  0.07076044,  0.08553714],
                 [ 0.01493013, -0.04381905, -0.13858535, -0.01954075, -0.01262037, -0.06094643],
                 [ 0.01181968, -0.13858535,  0.02152272, -0.0591145,   0.00783004, -0.00460452],
                 [ 0.14345572, -0.00168609, -0.04497956,  0.10727923,  0.00533928,  0.21749003],
                 [ 0.12730021, -0.01262037,  0.00783004,  0.06187905, -0.00783993,  0.01841808],
                 [ 0.10880262, -0.04309177,  0.00953042,  0.20209902, -0.03812169,  0.1008121 ]])
        self.__overlap__(2, 2, S_ref)


    def __nuclear_attraction__(self, l_a, l_b, V_ref):

        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])
        C = np.array([1.0, 0.0, 0.0])

        Z = 1.0
        alpha = 0.8
        beta = 1.1

        V = NucAttIntegralGTO(A, alpha, l_a, B, beta, l_b, C, Z)

        assert np.allclose(V.integral().flatten(), V_ref.flatten())


    def test_nuclear_attraction_ss(self):
        S_ref = np.array([[-0.66371193]])
        self.__nuclear_attraction__(0, 0, S_ref)

    def test_nuclear_attraction_ps(self):
        S_ref = np.array([[-0.22640975],
                         [ 0.52905935],
                         [ 0.38521547],])
        self.__nuclear_attraction__(1, 0, S_ref)


    def test_nuclear_attraction_ds(self):
        S_ref = np.array([[-0.24104726],
                          [ 0.3385889 ],
                          [ 0.23532124],
                          [-0.57017003],
                          [-0.53656878],
                          [-0.36365438],])
        self.__nuclear_attraction__(2, 0, S_ref)


    def test_nuclear_attraction_sd(self):
        S_ref = np.array([[-0.16493548,  0.00694757, -0.00975703, -0.25617277, -0.17589804, -0.230745  ]])
        self.__nuclear_attraction__(0, 2, S_ref)


    def test_nuclear_attraction_pp(self):
        S_ref = np.array([[-0.17312433, -0.07620731, -0.07922648],
                          [ 0.03676658,  0.06470119,  0.19281826],
                          [ 0.02029814,  0.15247044,  0.00230032],])
        self.__nuclear_attraction__(1, 1, S_ref)


    def test_nuclear_attraction_pd(self):
        S_ref = np.array([[-0.05297955, -0.11206175, -0.10823337, -0.06997963, -0.04857306, -0.06782339],
                          [ 0.13202034,  0.00356592,  0.01590478,  0.07178116,  0.04160932,  0.17727315],
                          [ 0.09589011, -0.00240384,  0.01169424,  0.14552154, -0.00544525,  0.02616033],])
        self.__nuclear_attraction__(1, 2, S_ref)

    def test_nuclear_attraction_dp(self):
        S_ref = np.array([[-0.10491685, -0.08756107, -0.08625651],
                          [ 0.24777042,  0.03718565,  0.11609609],
                          [ 0.17663377,  0.07682212,  0.01288952],
                          [-0.04206112, -0.00586027, -0.20735875],
                          [-0.04459273, -0.06485314, -0.01022637],
                          [-0.01253156, -0.14799073,  0.02397503],])
        self.__nuclear_attraction__(2, 1, S_ref)


    def test_nuclear_attraction_dd(self):
        S_ref = np.array([[-0.13298099, -0.05661908, -0.06223912, -0.08238682, -0.0568285,  -0.07690931],
                          [ 0.08711538,  0.04873719,  0.15176828,  0.0411425,   0.02293687,  0.09563935],
                          [ 0.05740601,  0.11120126,  0.00840554,  0.07010278,  0.00141246,  0.01817752],
                          [-0.14279632, -0.00043519, -0.01832593, -0.08853108, -0.00374837, -0.18998083],
                          [-0.13438062, -0.00455151, -0.02006412, -0.07192937, -0.00022514, -0.03778204],
                          [-0.09031926,  0.00473271, -0.00652531, -0.14209309,  0.02176417, -0.07062916],])
        self.__nuclear_attraction__(2, 2, S_ref)



    def __kinetic__(self, l_a, l_b, T_ref):

        A = np.array([0.0, 1.2, 0.7])
        B = np.array([0.3, 0.0, -0.25])

        alpha = 0.8
        beta = 1.1

        T = KineticIntegralGTO(A, alpha, l_a, B, beta, l_b)

        assert np.allclose(T.integral().flatten(), T_ref.flatten())

    def test_kinetic_ss(self):
        S_ref = np.array([[0.23834233]])
        self.__kinetic__(0, 0, S_ref)

    def test_kinetic_ps(self):
        S_ref = np.array([[ 0.15226882],
                          [-0.60907529],
                          [-0.48218461],])
        self.__kinetic__(1, 0, S_ref)


    def test_kinetic_pp(self):
        S_ref = np.array([[ 0.19747143,  0.13295563,  0.10525654],
                          [ 0.13295563, -0.30111217, -0.42102616],
                          [ 0.10525654, -0.42102616, -0.10260204],])
        self.__kinetic__(1, 1, S_ref)


    def test_kinetic_sd(self):
        S_ref = np.array([[-0.03527816, -0.16748066, -0.13258885,  0.3273281,   0.53035541,  0.18295708]])
        self.__kinetic__(0, 2, S_ref)


    def test_kinetic_dp(self):
        S_ref = np.array([[ 0.13032456,  0.03268356,  0.02587449],
                          [-0.42291284, -0.10745725, -0.18002328],
                          [-0.334806,   -0.18002328, -0.02257785],
                          [-0.13125357, -0.0289675,   0.41563631],
                          [-0.18002328,  0.34028128,  0.09031141],
                          [-0.08224843,  0.32899372, -0.17811555],])
        self.__kinetic__(2, 1, S_ref)


    def test_kinetic_dd(self):
        S_ref = np.array([[ 0.09604206,  0.14442149,  0.11433368, -0.00377303,  0.08819035, -0.02777982],
                          [ 0.10516821, -0.25480035, -0.42137641, -0.03447746, -0.08613901, -0.15114887],
                          [ 0.08325817, -0.42137641, -0.05612506, -0.16367593, -0.01250572,  0.01672175],
                          [ 0.08121281,  0.00477582, -0.13260042, -0.14199375, -0.01512341,  0.29946709],
                          [ 0.2124924 , -0.08613901, -0.01250572,  0.10917864, -0.0092286,  -0.06688698],
                          [ 0.02336906, -0.11189559,  0.04779726,  0.26563013, -0.19118903, -0.11966924],])
        self.__kinetic__(2, 2, S_ref)


    def test_read_basis(self):

        file_path = os.path.dirname(__file__)
        file_name = os.path.join(file_path, "single_atom_sto3g.txt")
        b = QchemBasis(file_name)

        assert b.n_aos == 5
        assert b.n_shells == 3

        centers = np.array(b.centers)
        centers_ref = np.array([0.0, 0.0, 0.0])

        l = np.array(b.angular_momentum)
        l_ref = np.array([0, 0, 1], dtype=int)

        exponents = np.array(b.exponents)
        exponents_ref = np.array([2.07015610000000e+02, 3.77081510000000e+01, 1.02052970000000e+01, 8.24631510000000e+00, 1.91626620000000e+00, 6.23229300000000e-01, 8.24631510000000e+00, 1.91626620000000e+00, 6.23229300000000e-01])

        coefficients = np.array(b.coefficients)
        coefficients_ref = np.array([6.00288292416504e+00, 5.80572680555135e+00, 1.80939142283201e+00,-3.46707063475341e-01, 4.63748691732151e-01, 3.49998076979097e-01, 3.10567802836512e+00, 1.95293364225592e+00, 3.09377532603190e-01])

        assert np.allclose(centers, centers_ref)
        assert np.allclose(l, l_ref)
        assert np.allclose(exponents.flatten(), exponents_ref.flatten())
        assert np.allclose(coefficients.flatten(), coefficients_ref.flatten())


    def test_overlap_basis(self):

        file_path = os.path.dirname(__file__)
        file_name = os.path.join(file_path, "single_atom_sto3g.txt")
        b = QchemBasis(file_name)

        S = OverlapGTO(b)
        I = S.get_overlap()

        I_ref = np.array([[1.,         0.24278166, 0.,         0.,         0.,       ],
                          [0.24278166, 1.,         0.,         0.,         0.,       ],
                          [0.,         0.,         1.,         0.,         0.,       ],
                          [0.,         0.,         0.,         1.,         0.,       ],
                          [0.,         0.,         0.,         0.,         1.,       ],])


        assert np.allclose(I, I_ref)


    def test_kinetic_basis(self):

        file_path = os.path.dirname(__file__)
        file_name = os.path.join(file_path, "single_atom_sto3g.txt")
        b = QchemBasis(file_name)

        T = KineticGTO(b)
        I = T.get_kinetic()

        I_ref = np.array([[45.93487167, -0.2574199,   0.,          0.,          0.        ],
                          [-0.2574199,   1.32403678,  0.,          0.,          0.        ],
                          [ 0.,          0.,          4.14307308,  0.,          0.        ],
                          [ 0.,          0.,          0.,          4.14307308,  0.        ],
                          [ 0.,          0.,          0.,          0.,          4.14307308],])


        assert np.allclose(I, I_ref)


    def test_nuclear_attraction_basis(self):

        file_path = os.path.dirname(__file__)
        file_name = os.path.join(file_path, "single_atom_sto3g.txt")
        b = QchemBasis(file_name)

        pc = [PointCharge(np.array(b.centers[0]), 1.0)]

        V = NucAttGTO(b, pc)
        I = V.get_nuclear_attraction()

        I_ref = np.array([[-9.53593255, -1.16012674,  0.,          0.,          0.,        ],
                          [-1.16012674, -1.44830695,  0.,          0.,          0.,        ],
                          [ 0.,          0.,         -1.43671274,  0.,          0.,        ],
                          [ 0.,          0.,          0.,         -1.43671274,  0.,        ],
                          [ 0.,          0.,          0.,          0.,         -1.43671274,],])


        assert np.allclose(I, I_ref)



