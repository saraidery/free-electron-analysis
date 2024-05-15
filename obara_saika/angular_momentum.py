import numpy as np
from scipy.special import factorial

def get_n_cartesian(l):
    return int(factorial(2 + l) / (factorial(l) * factorial(2)))

def get_cartesian_index(a):
    idx = 0

    for x in range(sum(a), -1, -1):
        for y in (range(sum(a) - x, -1, -1)):
            z = sum(a) - x - y
            if (x == a[0] and y == a[1] and z == a[2]):
                return idx
            idx += 1
    return idx

def get_cartesians(l):
    cart=[]

    for x in range(l, -1, -1):
        for y in (range(l - x, -1, -1)):
            z = l - x - y
            cart.append(np.array([x, y, z], dtype=int))

    return cart

def get_n_cartesian_accumulated(l):
    n = 0
    for l_ in np.arange(l + 1):
        n += get_n_cartesian(l_)
    return n

def get_cartesian_index_accumulated(a):
    idx = 0

    for l in np.arange(sum(a)):
        idx += get_n_cartesian(l)

    idx += get_cartesian_index(a)

    return idx

def get_cartesians_accumulated(l):
    cart=[]
    for l_ in np.arange(l+1):
        for a in get_cartesians(l_):
            cart.append(a)

    return cart
