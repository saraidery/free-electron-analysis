from basis_utilities import plot_GTO_basis, Basis
from even_tempered_basis import *
from QChem_solution import *

# Generate the basis (following should match the free electron test)
beta = 1.8
initial_alpha = 100.0
E =  0.02321
l = [0, 1, 2, 3]
N = [30, 15, 5, 1]

center = np.array([0.0, 0.0, 0.0])

coefficients = np.ones(sum(N))
exponents = []
angular_momentum = []
for i, n in enumerate(N):
    exponents += generate_even_tempered_from_least_diffuse(initial_alpha, n, beta)
    angular_momentum += [l[i]] * n


basis = Basis(coefficients, exponents, angular_momentum, center)



k = np.array([0.0, 0.0, np.sqrt(2.0*E)])

plot_GTO_basis("basis", basis)

for i in range(2):
    QChem_solution_file = f"/Users/sarai/Desktop/Qchem_solution_{i}.txt"
    plot_QC_PWGTO(k, QChem_solution_file, basis)
#plot_QC_GTO(QChem_solution_file, basis)