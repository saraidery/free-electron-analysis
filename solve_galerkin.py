import numpy as np
import os

from numpy import linalg as la
import matplotlib.pyplot as plt

from obara_saika import OverlapPWGTO, KineticPWGTO, NucAttPWGTO
from scipy.linalg.lapack import zggev, zhegv, zheev, zpstrf, ztrtri

from basis_utilities import *
from QChem_solution import *

def get_k_scale_factors(k, b):

    k_scale = np.zeros(b.n_aos)

    i = 0
    for shell in b.shells:
        l = shell.l
        for x in range(l, -1, -1):
            for y in (range(l - x, -1, -1)):
                z = l - x - y
                k_scale[i]+= np.power(k[0], x)*np.power(k[1], y)*np.power(k[2], z)
                i += 1
    return k_scale

def cd_basis_transformation_matrix(overlap):
    L, piv, rank, info = zpstrf(overlap, tol=1.0e-5,lower=1)
    L = np.tril(L[0:rank, 0:rank])

    P = np.zeros([np.shape(overlap)[0], rank])
    for i in range(rank):
        P[piv[i]-1, i] = 1.0

    L_inv, info = ztrtri(L, lower=1)

    X = np.dot(P, L_inv.T)
    print(f"Biggest error in Cholesky decomp: {np.max(np.abs(np.matmul(X.T, np.matmul(overlap, X)) - np.eye(rank)))}")

    return X, rank

def do_k_scaling(b, k, M):
    k_scaling = get_k_scale_factors(k, b)
    for i in range(b.n_aos):
        for j in range(b.n_aos):
            M[i, j] *= k_scaling[i]*k_scaling[j]
    return M

def solve_least_squares(b, E, center, Z, k):

    S = OverlapPWGTO(b)
    overlap = S.get_overlap()

    overlap = do_k_scaling(b, k, overlap)

    X, rank = cd_basis_transformation_matrix(overlap)

    print(f"Rank of S: {rank}")

    pc = [PointCharge(center, Z)]
    V = NucAttPWGTO(b, pc)
    H = V.get_nuclear_attraction()

    K = KineticPWGTO(b)
    H += K.get_kinetic()

    H = do_k_scaling(b, k, H)
    H = np.dot(X.T, np.dot(H, X)) # in orthonormal basis


    # Least squares problem
    A = H - E*np.eye(rank)
    A = np.matmul(A.conj().T, A)

    # Solve eigenvalue problem
    w, v, info = zheev(A, compute_v=True)

    # print("Errors:")
    # print(w)

    # Transform back to original basis
    C = np.matmul(X, v)

    return C, w

E = 0.5
k = np.array([0.0, 0.0, np.sqrt(2.0*E)])
Z = 1.0

QC_basis = QchemBasis("/Users/sarai/prog/libqints-ang-mom/libfree/tests/atoms_and_molecules/6-311gd2+/Ne.txt", k)

# Generate the basis (following should match the free electron test)
initial_alpha = 0.4
beta = 2.0
l = [0, 1, 2, 3, 4, 5, 6]
N = [5, 5, 5, 5, 5, 5, 5]

center = np.array([0.0, 0.0, 0.0])
centers = [center]

etb = EvenTemperedBasis(N, l, beta, initial_alpha, centers, True, k)
b = Basis(QC_basis.shells + etb.shells)

C, w = solve_least_squares(b, E, center, Z, k)

print(w)

fig, ax = plot_solution(k, C[:,0], b, k_scale_angular_momentum=True)
plt.savefig("solution.png")



