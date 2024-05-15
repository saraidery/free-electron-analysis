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


def coulomb_wave(k, Z, x_):

    CW_m = np.zeros(len(x_), dtype=complex)

    eta = -Z/k
    a = complex(0.0, eta)

    gamma_m = np.array(mp.gamma(complex(1.0, -a)), dtype=complex)

    HG_m = np.zeros(len(x_), dtype=complex)
    PW = np.zeros(len(x_), dtype=complex)

    for i, x in enumerate(x_):
        c = -2.0*k*x*1j
        val = mp.hyp1f1(a, 1.0, c)
        HG_m[i] = val
        PW[i] = complex(np.cos(k*x), np.sin(k*x))

    exp_eta = np.exp(-np.pi*eta/2.0)

    CW_m = gamma_m * exp_eta * np.multiply(PW, HG_m)


    return CW_m

def plot_CW(k_norm, Z, x_max=10, n_x=100):
    x_ = np.linspace(0, x_max, num=n_x)
    CW = coulomb_wave(k_norm, Z, x_)

    CW /= np.amax(np.abs(CW))

    CW_sq = np.imag(CW)*np.imag(CW) + np.real(CW)*np.real(CW)

    norm = np.amax(CW_sq)
    CW_sq /=norm

    fig, ax = plt.subplots(2,1)

    ax[0].plot(x_, np.real(CW), ':', color='C0', label=r"$Re(\Psi)$")
    ax[0].plot(x_, np.imag(CW), '--', color='C0', label=r"$Im(\Psi)$")
    ax[1].plot(x_, CW_sq, color='C0', label=r"$|\Psi|^2$")

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlim([0, x_max])
    ax[1].set_xlim([0, x_max])
    ax[1].set_ylim([0, 1.0])
    fig.tight_layout()
    plt.savefig("CW.png")
    plt.close()
