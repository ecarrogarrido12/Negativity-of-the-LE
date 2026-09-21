import numpy as np
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


# Physical parameters
alpha = 50  # Bifurcation parameter
L = 1  # Domain O = [0, L]

# Numerical parameters
N = 1000 # Number of grid points
K = 1  # Fourier series truncation
R = 2 * np.sqrt(alpha)  # Truncation of the real line for the Lebesgue integral
x = np.linspace(0, L, N) # Spatial domain grid
x_int = x[1:-1] # Interior points
dx = x[1] - x[0] # Mesh size

# First eigenfunction
lambda_1 = np.pow(np.pi / L, 2)
e_1 = np.sqrt(2.0 / L) * np.sin(np.pi * x_int / L)

# Orthonormal basis for H = L^2(O; R^2)
def e_(k):
    return np.sqrt(2.0 / L) * np.sin(k * np.pi * x / L)

# Eigenvalues of the orthonormal basis
def lambda_(k):
    return np.pow(k * np.pi / L, 2)

# Sample f from N(0, (-Delta)^-1) measure
def f_mc(xi):
    sample = np.zeros((2, len(x)))
    for k in range(1, K + 1):
        sample[0] += (xi[k - 1][0] / np.sqrt(lambda_(k))) * e_(k)
        sample[1] += (xi[k - 1][1] / np.sqrt(lambda_(k))) * e_(k)
    return sample

def potential(r):
    return (
            -alpha / (2 * lambda_1) * r ** 2
            + 3 / (8 * L * lambda_1 ** 2) * r ** 4
    )


def le_schrodinger_op(r):
    # Finite difference Laplacian
    main = -2 * np.ones(N - 2)
    off = np.ones(N - 3)
    delta = diags(
        [off, main, off],
        [-1, 0, 1],
        shape=(N - 2, N - 2)
    ) / dx ** 2

    # Schrodinger operator
    f_squared = (r ** 2 / lambda_1) * e_1 ** 2
    l_f = delta + diags(alpha - f_squared, 0)

    # Compute the largest eigenvalue of L_f
    return eigsh(l_f, k=1, which='LA', return_eigenvectors=False)[0]


# Integrands for the expectation
def numerator_integrand(r):
    return le_schrodinger_op(r) * np.exp(-0.5 * r ** 2 - 2 * potential(r))* r

def denominator_integrand(r):
    return np.exp(-0.5 * r**2 - 2 * potential(r)) * r

if __name__ == "__main__":
    numerator = quad(
        numerator_integrand,
        16,
        19,
        limit=200
    )[0]

    denominator = quad(
        denominator_integrand,
        16,
        19,
        limit=200
    )[0]

    expectation = numerator / denominator
    print(expectation)