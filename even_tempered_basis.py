import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial2

angular_momentum_dict =  {
  "S": 0,
  "P": 1,
  "D": 2,
  "F": 3,
  "G": 4,
  "H": 5,
  "I": 6,
}



def get_n_cartesian(l):
    return int(factorial(2 + l) / (factorial(l) * factorial(2)))

def get_n_pure(l):
    return 2*l + 1

def generate_even_tempered_from_least_diffuse(exp_max, k_max, beta):

    exponents = [exp_max]
    for k in np.arange(k_max-1):
        exponents.append(exponents[-1]/beta)
    return exponents

def generate_even_tempered_from_most_diffuse(exp_min, k_max, beta):

    exponents = [exp_min]
    for k in np.arange(k_max-1):
        exponents.append(exponents[-1]*beta)

    return exponents

def get_min(l, r_max, limit_value=0.5):
    return -np.log(limit_value/pow(r_max, l))/pow(r_max,2)

def get_beta(min_val, max_val, N):
    if N == 1:
        return 1.0
    return pow(max_val/min_val, float(1/(N-1)))

def get_N(min_val, max_val, beta):
    alpha = min_val/beta
    log_ = np.log(max_val*beta/min_val)
    log_beta = np.log(beta)

    return int(log_/log_beta)

