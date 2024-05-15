import numpy as np
import os

from numpy import linalg as la
import matplotlib.pyplot as plt

from obara_saika import OverlapPWGTO, KineticPWGTO, NucAttPWGTO
from scipy.linalg.lapack import zggev, zhegv, zheev

from basis_utilities import *

E = 5.0*0.00367
k = np.array([0.0, 0.0, np.sqrt(2.0*E)])
Z = 1.0

# Generate the basis (following should match the free electron test)
initial_alpha = 10.0
beta = 1.5
l = [0, 1]
N = [2, 2]

center_1 = np.array([0.0, 0.0, 0.0])
centers = [center_1]

b = EvenTemperedBasis(N, l, beta, initial_alpha, centers, False, k)
S = OverlapPWGTO(b)
I = S.get_overlap()
print(I)


pc = [PointCharge(centers[0], Z)]
V = NucAttPWGTO(b, pc)
I = V.get_nuclear_attraction()
print(I)

K = KineticPWGTO(b)
I = K.get_kinetic()
print(I)

