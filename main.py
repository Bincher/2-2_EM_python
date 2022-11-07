import numpy as np
from scipy.linalg import lu
from numpy.linalg import inv

def lu(A):
    # Get the number of rows
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n, dtype=np.double)

    # Loop over rows
    for i in range(n):
        # Eliminate entries below i with row operations
        # on U and reverse the row operations to
        # manipulate L
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return L, U


def plu(A):
    # Get the number of rows
    n = A.shape[0]

    # Allocate space for P, L, and U
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)

    # Loop over rows
    for i in range(n):

        # Permute rows if needed
        for k in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k + 1]] = U[[k + 1, k]]
            P[[k, k + 1]] = P[[k + 1, k]]

        # Eliminate entries below i with row
        # operations on U and #reverse the row
        # operations to manipulate L
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return P, L, U


A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]])
b = np.array([[7.85], [-19.3], [71.4]])

p, l, u = plu(A)

print("l")
print(l)

print("u")
print(u)

print("l * u (x)")
print(l*u)

print("l * u (o)")
print(np.dot(l, u))

d = np.dot(inv(l), b)
print("전진대입")
print(d)

x = np.dot(inv(u), d)
print("후진대입")
print(x)