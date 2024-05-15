from basis_utilities import plot_GTO_basis, Basis
from even_tempered_basis import *
from QChem_solution import *

import matplotlib.pyplot as plt

# Generate the basis (following should match the free electron test)
initial_alpha_1 = 1000.0
beta_1 = 1.5
l_1 = [0, 1, 2, 3, 4, 5, 6]
N_1 = [15, 15, 15, 15, 15, 15, 15]

initial_alpha_2 = 30.0
beta_2 = 1.75
l_2 = [0, 1, 2, 3, 4, 5, 6]
N_2 = [10, 10, 10, 10, 10, 10, 10]

initial_alpha_3 = 1.0
beta_3 = 2.0
l_3 = [0, 1, 2, 3, 4]
N_3 = [10, 10, 10, 10, 10]

center_1 = np.array([0.0, 0.0, 0.0])
centers = [center_1]

ET_basis_1 = EvenTemperedBasis(N_1, l_1, beta_1, initial_alpha_1, centers, False)
ET_basis_2 = EvenTemperedBasis(N_2, l_2, beta_2, initial_alpha_2, centers, True)
ET_basis_3 = EvenTemperedBasis(N_3, l_3, beta_3, initial_alpha_3, centers, True)

#QC_basis = QchemBasis("/Users/sarai/prog/libqints-ang-mom/libfree/tests/atoms_and_molecules/6-311gd2+/Ne.txt")
#basis = Basis(QC_basis.shells + ET_basis.shells)

basis = Basis(ET_basis_1.shells + ET_basis_2.shells + ET_basis_3.shells)

# plot_GTO_basis("et_basis_1", ET_basis_1, x_max=10, n_x=100)
# plot_GTO_basis("et_basis_2", ET_basis_2, x_max=10, n_x=100)
# plot_GTO_basis("et_basis_3", ET_basis_3, x_max=10, n_x=100)
# plot_GTO_basis("basis", basis, x_max=10, n_x=100)

E = 5.0*0.00367
k = np.array([0.0, 0.0, np.sqrt(2.0*E)])
Z = 1.0

plot_CW(np.sqrt(2.0*E), Z, x_max=10, n_x=100)

# for shell in basis.shells:
#     print(shell)

for i in range(1):
    QChem_solution_file = f"/Users/sarai/Desktop/Free-electron-tests/Qchem_solution_{i}.txt"
    print( QChem_solution_file)
    plot_QC_PWGTO(k, QChem_solution_file, basis, True)