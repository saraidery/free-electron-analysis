import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from basis_utilities import *
from cw import *

def plot_QC_GTO(QChem_solution, gto):

    prepare_QC_file(QChem_solution)

    coefficients = np.loadtxt(QChem_solution, skiprows=1).view(complex)

    x_max = 30
    n_x = 500

    x_ = np.linspace(0.0, x_max, num=n_x)
    r = np.zeros(3, dtype=float)
    y = np.zeros([n_x, gto.n_aos()],dtype=complex)

    for i, x in enumerate(x_):
        r[0] = x
        y[i,:] = evaluate_GTOs_at_points(r, gto)

    z = np.dot(y,coefficients)
    z /= np.amax(np.abs(z))
    z_sq = np.imag(z)*np.imag(z) + np.real(z)*np.real(z)

    norm = np.amax(z_sq)
    z_sq /=norm

    fig, ax = plt.subplots()
    ax.plot(x_, np.real(z), label=r"$|\Psi|$")
    ax.axhline(y=0.0,color='k')
    ax.axvline(x=0.0,color='k')

    ax.legend()
    ax.set_xlim([0.0,x_max])
    fig.tight_layout()
    plt.savefig(QChem_solution.replace(".txt", ".png"))
    plt.close()


def prepare_QC_file(QChem_solution):

        with open(QChem_solution, "r") as f:
            lines = f.readlines()

        with open(QChem_solution, "w") as f:
            for line in lines:
                line = line.replace("(", " ")
                line = line.replace(")", " ")
                line = line.replace(",", " ")
                f.write(line)
        return

def plot_QC_PWGTO(k, QChem_solution, gto):

    prepare_QC_file(QChem_solution)
    coefficients = np.loadtxt(QChem_solution, skiprows=1).view(complex)

    x_max = 30
    n_x = 500

    x_ = np.linspace(0, x_max, num=n_x)
    r = np.zeros(3, dtype=float)
    y = np.zeros([n_x, gto.n_aos()],dtype=complex)
    pw = np.zeros([n_x],dtype=complex)

    cw = coulomb_s_wave(k, 1.0, x_)

    for i, x in enumerate(x_):
        r[2] = x
        y[i,:] = evaluate_PWGTOs_at_points([k], r, gto)
        pw[i] = plane_wave(k, r)


    z = np.dot(y,coefficients)
    z /= np.amax(np.abs(z))
    # idx = np.argmax(np.abs(z))
    # if (np.real(z)[idx] < 0):
    #     z *= -1

    z_sq = np.imag(z)*np.imag(z) + np.real(z)*np.real(z)

    norm = np.amax(z_sq)
    z_sq /=norm


    cw /= np.amax(np.abs(cw))
    cw_sq = np.imag(cw)*np.imag(cw) + np.real(cw)*np.real(cw)
    norm = np.amax(cw_sq)
    cw_sq /=norm

    fig, ax = plt.subplots()
    ax.plot(x_, np.real(z), ':', color='C0', label=r"$Re(\Psi)$")
    ax.plot(x_, np.imag(z), '--', color='C0', label=r"$Im(\Psi)$")

    #ax.plot(x_, z_sq, color='C0', label=r"$|\Psi|^2$")
    #ax.plot(x_, cw_sq, color='C3', label=r"$|\Psi_{CW}|^2$")

    ax.plot(x_, np.real(cw), ':', color='C3', label=r"$Re(CW)$")
    ax.plot(x_, np.imag(cw), '--', color='C3', label=r"$Im(CW)$")
    #ax.plot(x_, np.real(pw), label=r"$Re(\exp(ikr))$")
    ax.axhline(y=0.0,color='k')
    ax.axvline(x=0.0,color='k')

    ax.legend()
    ax.set_xlim([0, x_max])
    fig.tight_layout()
    plt.savefig(QChem_solution.replace(".txt", ".png"))
    plt.close()