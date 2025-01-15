
import numpy as np
from obara_saika.angular_momentum import get_n_cartesian
from obara_saika.GTO import GTO

def kinetic_action(grid, gto):

    fn_on_grid = np.zeros_like(grid)

    for idx, r in enumerate(grid):
        fn_on_grid[idx] = -gto.alpha * gto.evaluate(r)

    for i, a in enumerate(gto.a):
        a_up = gto.a

        a_up[i] += 2
        gto_up = GTO(gto.alpha, gto.A, a_up)

        for idx, r in enumerate(grid):
            fn_on_grid[idx] += 2.0 * gto.alpha * gto_up.evaluate(r)

        if (a >= 2 ):
            a_down = gto.a
            a_down[i] -= 2
            gto_down = GTO(gto.alpha, gto.A, a_down)

            for idx, r in enumerate(grid):
                fn_on_grid[idx] += 0.5 * gto.a[i]*(gto.a[i]-1) * gto_down.evaluate(r)

        return fn_on_grid

def momentum_action(grid, gto):

    fn_on_grid = np.zeros_like(grid)
    vec_on_grid = np.vstack(fn_on_grid, fn_on_grid, fn_on_grid)

    for i in np.arange(3):

        a_up = gto.a
        a_up[i] += 1
        gto_up = GTO(gto.alpha, gto.A, a_up)

        for k, r in enumerate(grid):
            vec_on_grid[i, k] += 2.0 * gto.alpha * gto_up.evaluate(r)

        if (gto.a[i] >= 1 ):
            a_down = gto.a
            a_down[i] -= 1

            gto_down = GTO(gto.alpha, gto.A, a_down)

            for k, r in enumerate(grid):
                vec_on_grid[i, k] -= gto.a[i] * gto_down.evaluate(r)

    vec_on_grid = vec_on_grid * 1j
    return vec_on_grid

def coulomb_action(grid, gto, Z):

    fn_on_grid = np.zeros_like(grid)

    for idx, r in enumerate(grid):
        r_norm = np.sqrt(np.dot(r, r))
        fn_on_grid[idx] = gto.evaluate(r) * Z / r_norm

    return fn_on_grid

def spherical_grid_in_cart(n_phi, n_theta, n_r, r_max):

    phi = np.linspace(0, 2*np.pi, num=n_phi, endpoint=False)
    theta = np.linspace(0, np.pi, num=n_theta, endpoint=False)

    r, step = np.linspace(0, r_max, n_r, retstep=True)
    r = r + step

    n_g = n_r * n_phi * n_theta
    grid = np.zeros([n_g, 3])

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    z = np.ndarray.flatten(np.outer(np.outer(r, cos_theta), np.ones_like(phi)))
    x = np.ndarray.flatten(np.outer(np.outer(r, sin_theta), cos_phi))
    y = np.ndarray.flatten(np.outer(np.outer(r, sin_theta), sin_phi))

    grid[:,0] = x
    grid[:,1] = y
    grid[:,2] = z

    new_grid = [list(i) for i in grid]
    #using the .unique() with axis=0
    new_grid = np.unique(new_grid, axis=0)

    return new_grid



def test_grid():
    n_r = 10
    n_phi = 10
    n_theta = 10
    r_max = 5.0

    grid = spherical_grid_in_cart(n_phi, n_theta, n_r, r_max)
    print(np.shape(grid)[0])


def test_evaluation_at_point():
    r = np.array([0.1, 0.2, 0.3])

    A = np.zeros(3)
    alpha = 0.5
    a = np.array([1, 0, 0])

    gto = GTO( alpha,  A, a)

    print(gto.evaluate(r))


def test_evaluation_grid():
    n_r = 10
    n_phi = 10
    n_theta = 10
    r_max = 5.0

    grid = spherical_grid_in_cart(n_phi, n_theta, n_r, r_max)

    A = np.zeros(3)
    alpha = 0.5
    a = np.array([0, 0, 0])

    gto = GTO( alpha,  A, a)

    gto_on_grid = np.zeros(np.shape(grid)[0])

    for idx in np.arange(np.shape(grid)[0]):
        gto_on_grid[idx] = gto.evaluate(grid[idx,:])

    print(gto_on_grid)


test_grid()
test_evaluation_at_point()
test_evaluation_grid()


