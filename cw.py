import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.special import factorial2
from basis_utilities import *

def eta(k, Z):
	k_norm = np.sqrt(np.dot(k,k))
	return -Z/k_norm

def plane_wave_norm(k):
	k_norm = np.sqrt(np.dot(k, k))
	return np.sqrt(k_norm/pow(2.0*np.pi,3))

def hypergeometric_1F1(k, r, Z):
	kr = np.dot(k, r)
	k_norm = np.sqrt(np.dot(k,k))
	r_norm = np.sqrt(np.dot(r,r))

	a = complex(0.0, eta(k,Z))
	b = complex(1.0, 0.0)
	c = complex(0.0, -k_norm*r_norm-kr)
	return mp.hyp1f1(a, b, c)

def coulomb_prefactor(k, Z):
	k_norm = np.sqrt(np.dot(k,k))
	gamma = mp.gamma(complex(1.0, -eta(k,Z)))

	prefactor = gamma*plane_wave_norm(k)*np.exp(-np.pi*eta(k,Z)/2.0)
	return prefactor


def coulomb_s_wave(k, Z, x_):

	CW = np.zeros(x_.size, dtype=complex)
	for i, x in enumerate(x_):
		r = np.zeros(3)
		r[0] = x
		CW[i] = coulomb_prefactor(k, Z)*plane_wave(k, r)*hypergeometric_1F1(k, r, Z)
	return CW
