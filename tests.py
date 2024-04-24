from basis_utilities import plot_GTO_basis, Basis
from even_tempered_basis import *
from QChem_solution import *

# Generate the basis (following should match the free electron test)
N = 20
beta = 2.0
initial_alpha = 1000
E =  0.002312

l = [0, 1]
center = np.array([0.0, 0.0, 0.0])

coefficients = np.ones(N)
exponents = generate_even_tempered_from_least_diffuse(initial_alpha, N, beta)
print(exponents)
print(coefficients)
print(l)
print(center)
basis = Basis(coefficients, exponents, l, center)

QChem_solution_file = "/Users/sarai/Desktop/Qchem_solution.txt"

k = np.array([0.0, 0.0, np.sqrt(2.0*E)])

plot_GTO_basis("basis", basis)
plot_QC_PWGTO(k, QChem_solution_file, basis)
#plot_QC_GTO(QChem_solution_file, basis)