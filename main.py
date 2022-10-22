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

    print(A)
    print(b)

    x = np.zeros(n)
    k = n-1
    x[k-1] = A[k-1][k] / A[k-1][k-1]

    for i in range(k, -1, -1):
        x[i] = A[i, n] - A[i, i+1:n] * x[i + 1:n] / A[i, i]

    return x

A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3],[0.3, -0.2, 10]])
b = np.array([[7.85], [-19.3], [71.4]])

a1 = np.array([[0,-6,5],[0,2,7],[-4,3,-7]])
b1 = np.array([[50],[-30],[50]])
aa = np.array([[6,-1],[12,7],[-5,3]])
bb = np.array([[4,0],[0.6,8]])
cc = np.array([[1,-2],[-6,1]])
d1 = np.array([[120,-40,0,0,0],[-40,110,-70,0,0],[0,-70,170,-100,0],[0,0,-100,120,-20],[0,0,0,-20,20]])
e1 = np.array([[637.65],[735.75],[588.60],[735.75],[882.90]])







#x = gauss_naive(A,b)
#y = gauss_pivot(A,b)
#print(x)
#print(y)

#x1 = np.dot(np.linalg.inv(a1),b1)
#print(x1)
#print(np.transpose(a1))
#print(np.linalg.inv(a1))

#print("aa*bb:\n",np.dot(aa,bb))
#print("aa*cc:\n",np.dot(aa,cc))
#print("bb*cc:\n",np.dot(bb,cc))
#print("cc*bb:\n",np.dot(cc,bb))

x2 = np.dot(np.linalg.inv(d1),e1)
print(x2)