from basis_utilities import plot_GTO_basis, Basis
from even_tempered_basis import *
from QChem_solution import *

# Generate the basis (following should match the free electron test)
beta = 2.0
initial_alpha = 100.0
l = [0, 1, 2]#, 3# 4, 5, 6]
N = [30, 25, 20]#, 5, 5, 5, 5]

center_1 = np.array([0.0, 0.0, 0.0])
centers = [center_1]

ET_basis = EvenTemperedBasis(N, l, beta, initial_alpha, centers, True)
QC_basis = QchemBasis("/Users/sarai/prog/libqints-ang-mom/libfree/tests/atoms_and_molecules/6-311gd2+/Ne.txt")
#basis = Basis(QC_basis.shells + ET_basis.shells)

basis = Basis(ET_basis.shells)

plot_GTO_basis("basis", basis, x_max=10, n_x=100)

E =  0.5
k = np.array([0.0, 0.0, np.sqrt(2.0*E)])

for i in range(1):
    QChem_solution_file = f"/Users/sarai/Desktop/Free-electron-tests/Qchem_solution_{i}.txt"
    plot_QC_PWGTO(k, QChem_solution_file, basis, True)