import numpy as np
from scipy.linalg import lu
from numpy.linalg import inv, cholesky

def gauss_naive(A, b):

    n = len(A)
    if b.size != n:
        raise ValueError("Matrix A must be square", b.size, n)

    for pivot_row in range(n-1):
        for row in range(pivot_row + 1, n):
            factor = A[row][pivot_row]/A[pivot_row][pivot_row]

            A[row][pivot_row] = factor
            for col in range(pivot_row + 1, n):
                A[row][col] = A[row][col] - factor * A[pivot_row][col]

            b[row] = b[row] - factor * b[pivot_row]

    x = np.zeros(n)
    k = n-1
    x[k-1] = A[k-1][k] / A[k-1][k-1]

    for i in range(k, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:],x[i+1:]))/A[i,i]

    return x

def gauss_pivot(A, b):

    n = len(A)
    if b.size != n:
        raise ValueError("Matrix A must be square", b.size, n)

    for k in range(n-1):
        maxindex = abs(A[k: ,k]).argmax() + k
        if A[maxindex, k] == 0:
            raise ValueError("Matrix is singular.")

        if maxindex != k:
            A[[k, maxindex]] = A[[maxindex, k]]
            b[[k, maxindex]] = b[[maxindex, k]]

        for row in range(k+1, n):
            factor = A[row,k]/A[row,k]

            A[row][k] = factor
            for col in range(k + 1, n):
                A[row][col] = A[row][col] - factor*A[k][col]
            b[row] = b[row] - factor*b[k]
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

"""
A = np.array([[7.0, 2.0, -3.0], [2.0, 5.0, -3.0], [1.0, -1.0, -6.0]])
b = np.array([[-12.0], [-20.0], [-26.0]])


p, l, u = plu(A)


print("l")
print(l)
print("u")
print(u)
print("L*U")
print(np.dot(l,u))
"""

"""
print("l")
print(l)

print("u")
print(u)

print("l * u (o)")
print(np.dot(l, u))

b2 = np.array([12.0, 18.0, -6.0])
d = np.dot(inv(l), np.transpose(b2))
print("전진대입")
print(d)

x = np.dot(inv(u), d)
print("후진대입")
print(x)
"""

"""
A = np.array([[8.0, 20.0, 16.0],[20.0, 80.0, 50.0], [16.0, 50.0, 60.0]])
b = np.array([[100.0], [250.0], [100.0]])

print("np.array([[np.sum(A[0])], [np.sum(A[1])], [np.sum(A[2])]])")
print(b)

U = cholesky(A)
#print("U")
#print(U)

print("cholesky(A) -> U.T")
print(U.T)

print("np.dot(U, U.T)")
print(np.dot(U, U.T))

d = np.dot(inv(U),b)
print("np.dot(inv(U,b))")
print(d)

x = np.dot(inv(U.T), d)
print("np.dot(inv(U.T), d)")
print(x)
"""


A = np.array([[2.0, -1.0, 0.0],[-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])

U = cholesky(A)
#print("U")
#print(U)

print("A")
print(A)

print("cholesky(A) -> U.T")
print(U.T)

print("np.dot(U, U.T)")
print(np.dot(U, U.T))

"""
A = np.array([[9.0, 0.0, 0.0],[0.0, 25.0, 0.0], [0.0, 0.0, 16.0]])

U = cholesky(A)
#print("U")
#print(U)

print("cholesky(A) -> U.T")
print(U.T)
"""