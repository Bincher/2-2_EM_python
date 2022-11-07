import numpy as np
from scipy.linalg import lu
from numpy.linalg import inv, cholesky

A = np.array([[6, 15, 55],[15, 55, 225], [55, 225, 979]])
b = np.array([[np.sum(A[0])], [np.sum(A[1])], [np.sum(A[2])]])

print("np.array([[np.sum(A[0])], [np.sum(A[1])], [np.sum(A[2])]])")
print(b)

U = cholesky(A)
print("cholesky(A) -> U.T")
print(U.T)

print("np.dot(U, U.T)")
print(np.dot(U, U.T))

d = np.dot(inv(U),b)
print("np.dot(inv(U,b))")
print(d)

x = np.dot(inv(U.T), d)
print("np.dot(inv(U.T), d)")
print(x);