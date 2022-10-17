import numpy as np

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

A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3],[0.3, -0.2, 10]])
b = np.array([[7.85], [-19.3], [71.4]])

x = gauss_naive(A,b)
print(x)