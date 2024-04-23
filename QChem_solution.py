import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from basis_utilities import *

def plot_QC_GTO(QChem_solution, gto):

    coefficients = np.loadtxt(QChem_solution, skiprows=0).view(complex)

    x_max = 30
    n_x = 500

    x_ = np.linspace(-x_max, x_max, num=n_x)
    r = np.zeros(3, dtype=float)
    y = np.zeros([n_x, basis.n_aos()],dtype=complex)

    for i, x in enumerate(x_):
        r[0] = x
        y[i,:] = evaluate_GTOs_at_points(r, gto)

    z = np.dot(y,coefficients)
    z /= np.amax(np.abs(z))
    z_sq = np.imag(z)*np.imag(z) + np.real(z)*np.real(z)

    norm = np.amax(z_sq)
    z_sq /=norm

    fig, ax = plt.subplots()
    ax.plot(x_, z_sq, label=r"$|\Psi|^2$")
    ax.axhline(y=0.0,color='k')
    ax.axvline(x=0.0,color='k')

    ax.legend()
    ax.set_xlim([-x_max,x_max])
    fig.tight_layout()
    plt.savefig(QChem_solution.replace(".txt", ".png"))
    plt.close()


def plot_QC_PWGTO(E, QChem_solution, gto):

    coefficients = np.loadtxt(QChem_solution, skiprows=0).view(complex)

    x_max = 30
    n_x = 500

    x_ = np.linspace(-x_max, x_max, num=n_x)
    r = np.zeros(3, dtype=float)
    y = np.zeros([n_x, basis.n_aos()],dtype=complex)

    k = np.array([0.0, 0.0, 1.0])
    k = np.sqrt(2.0*E)*k

    for i, x in enumerate(x_):
        r[0] = x
        y[i,:] = evaluate_PWGTOs_at_points(k, r, gto)

    z = np.dot(y,coefficients)
    z /= np.amax(np.abs(z))
    z_sq = np.imag(z)*np.imag(z) + np.real(z)*np.real(z)

    norm = np.amax(z_sq)
    z_sq /=norm

    fig, ax = plt.subplots()
    ax.plot(x_, z_sq, label=r"$|\Psi|^2$")
    ax.axhline(y=0.0,color='k')
    ax.axvline(x=0.0,color='k')

    ax.legend()
    ax.set_xlim([-x_max,x_max])
    fig.tight_layout()
    plt.savefig(QChem_solution.replace(".txt", ".png"))
    plt.close()