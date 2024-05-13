from basis_utilities import plot_GTO_basis, Basis
from even_tempered_basis import *
from QChem_solution import *

# Generate the basis (following should match the free electron test)
initial_alpha = 100.0
beta = 1.5
l = [0, 1, 2, 3, 4]
N = [20, 20, 20, 20, 20]

center_1 = np.array([0.0, 0.0, 0.0])
centers = [center_1]

ET_basis = EvenTemperedBasis(N, l, beta, initial_alpha, centers, False)
QC_basis = QchemBasis("/Users/sarai/prog/libqints-ang-mom/libfree/tests/atoms_and_molecules/6-311gd2+/Ne.txt")
#basis = Basis(QC_basis.shells + ET_basis.shells)

basis = Basis(ET_basis.shells)

plot_GTO_basis("basis", basis, x_max=10, n_x=100)

for shell in basis.shells:
    print(shell)

E = 0.5
k = np.array([0.0, 0.0, np.sqrt(2.0*E)])

# QChem_solution_file = f"/Users/sarai/Desktop/Free-electron-tests/Qchem_solution.txt"
# plot_QC_PWGTO(k, QChem_solution_file, basis, True)