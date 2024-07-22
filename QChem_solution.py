import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from basis_utilities import *
from cw import *


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

def plot_QC_PWGTO(k, QChem_solution, gto, k_scale_angular_momentum=False):

    prepare_QC_file(QChem_solution)
    coefficients = np.loadtxt(QChem_solution, skiprows=0).view(complex)

    fig, ax = plot_solution(k, coefficients, gto, k_scale_angular_momentum)

    plt.savefig(QChem_solution.replace(".txt", ".png").split("/")[-1])

def plot_solution(k, C, gto, k_scale_angular_momentum=False):
    x_max = 10
    n_x = 500

    # Generalize the following lines so that r  and k point in the same direction
    x_ = np.linspace(0, x_max, num=n_x)
    r = np.zeros(3, dtype=float)
    y = np.zeros([n_x, gto.n_aos],dtype=complex)

    for i, x in enumerate(x_):
        r[2] = x
        y[i,:] = evaluate_PWGTOs_at_points([k], r, gto, k_scale_angular_momentum) # Call should be general for any direction of r/k


    z = np.dot(y,C)
    z /= np.amax(np.abs(z))
    z_sq = np.imag(z)*np.imag(z) + np.real(z)*np.real(z)

    norm = np.amax(z_sq)
    z_sq /=norm

    fig, ax = plt.subplots(2,1)
    ax[0].plot(x_, np.real(z), ':', color='C0', label=r"$Re(\Psi)$")
    ax[0].plot(x_, np.imag(z), '--', color='C0', label=r"$Im(\Psi)$")
    ax[1].plot(x_, z_sq, color='C0', label=r"$|\Psi|^2$")

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlim([0, x_max])
    ax[1].set_xlim([0, x_max])
    ax[1].set_ylim([0, 1.0])
    fig.tight_layout()

    return fig, ax

