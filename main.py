import numpy as np
from numpy import linalg as LA

def seidel(a, x, b):
    # Finding length of a(3)
    n = len(a)
    # for loop for 3 times as to calculate x, y , z
    for j in range(0, n):
        # temp variable d to store b[j]
        d = b[j]

        # to calculate respective xi, yi, zi
        for i in range(0, n):
            if (j != i):
                d -= a[j][i] * x[i]
        # updating the value of our solution
        x[j] = d / a[j][j]
    # returning our updated solution
    return x

def gaussSeidel(A, b, x, N, tol):
    maxIterations = 1000000
    xprev = [0.0 for i in range(N)]
    for i in range(maxIterations):
        for j in range(N):
            xprev[j] = x[j]
        for j in range(N):
            summ = 0.0
            for k in range(N):
                if (k != j):
                    summ = summ + A[j][k] * x[k]
            x[j] = (b[j] - summ) / A[j][j]
        diff1norm = 0.0
        oldnorm = 0.0
        for j in range(N):
            diff1norm = diff1norm + abs(x[j] - xprev[j])
            oldnorm = oldnorm + abs(xprev[j])
        if oldnorm == 0.0:
            oldnorm = 1.0
        norm = diff1norm / oldnorm
        if (norm < tol) and i != 0:
            print("Sequence converges to [", end="")
            for j in range(N - 1):
                print(x[j], ",", end="")
            print(x[N - 1], "]. Took", i + 1, "iterations.")
            return
    print("Doesn't converge.")

"""
a = [[6.0, -1.0], [3.0, 8.0]]
b = [5.0, 11.0]
x = [0.0, 0.0]

for i in range(0, 3):
    x = seidel(a, x, b)
    #print each time the updated solution
    print(x)
    print("et1 = ", (abs(1 - x[0]) / 1) * 100, "%")
    print("et2 = ", (abs(1 - x[1]) / 1) * 100, "%")
"""

"""
a = [[10.0, 2.0, -1.0], [-3.0, -6.0, 2.0], [1.0, 1.0, 5.0]]
b = [27.0, -61.5, -21.5]
x = [0.0, 0.0, 0.0]

gaussSeidel(a, b, x, 3, 5)
"""