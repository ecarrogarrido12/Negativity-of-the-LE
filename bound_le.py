import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# Physical parameters
alpha = 25  # Bifurcation parameter
L = 1  # Domain O = [0, L]

# Numerical parameters
N = 1000 # Number of grid points
K = 100 # Fourier series truncation
S = 10000 # MC samples
x = np.linspace(0, L, N) # Spatial domain grid
x_int = x[1:-1] # Interior points
dx = x[1] - x[0] # Mesh size

# First eigenfunction
lambda_1 = np.pow(np.pi / L, 2)
e_1 = np.sqrt(2.0 / L) * np.sin(np.pi * x_int / L)

# Orthonormal basis for H = L^2(O; R^2)
def e_(k):
    return np.sqrt(2.0 / L) * np.sin(k * np.pi * x_int / L)

# Eigenvalues of the orthonormal basis
def lambda_(k):
    return np.pow(k * np.pi / L, 2)

# Sample f from N(0, (-Delta)^-1) measure
def f_mc(xi):
    sample = np.zeros((2, N - 2))
    for k in range(1, K + 1):
        sample[0] += (xi[k - 1][0] / np.sqrt(lambda_(k))) * e_(k)
        sample[1] += (xi[k - 1][1] / np.sqrt(lambda_(k))) * e_(k)
    return sample

def potential(xi):
    f = f_mc(xi)
    f_squared = f[0]**2 + f[1]**2
    l2_squared = 0.0
    for k in range(1, K + 1):
        l2_squared += (xi[k - 1][0]**2 + xi[k - 1][1]**2)/lambda_(k)
    l4_forth = np.trapezoid(f_squared**2, x_int)
    return -alpha / 2 * l2_squared + 1 / 4 * l4_forth


def le_schrodinger_op(xi):
    # Finite difference Laplacian
    main = -2 * np.ones(N - 2)
    off = np.ones(N - 3)
    delta = diags(
        [off, main, off],
        [-1, 0, 1],
        shape=(N - 2, N - 2)
    ) / dx ** 2

    # Schrodinger operator
    f = f_mc(xi)
    f_squared = f[0]**2 + f[1]**2
    l_f = delta + diags(alpha - f_squared, 0)

    # Compute the largest eigenvalue of L_f
    return eigsh(l_f, k=1, which='LA', return_eigenvectors=False)[0]


if __name__ == "__main__":
    les = np.zeros(S)
    log_weights = np.zeros(S)
    for s in range(S):
        xi_array = np.random.uniform(-1, 1, (K, 2))
        les[s] = le_schrodinger_op(xi_array)
        log_weights[s] = -0.5 * np.sum(xi_array**2) - 2 * potential(xi_array)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    expectation = (np.sum(weights * les)/ np.sum(weights))
    print(expectation)



