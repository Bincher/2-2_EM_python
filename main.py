from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import numpy as np

def Newint(x, y, xx):
    n = len(x)
    if (len(y) != n):
        print("ERROR : x and y must be same length")
        return
    b = np.zeros((n,n))
    b[:,1] = y[:]
    for j in range(2,n):
        for i in range(1, n-j+1):
            b[i,j] = (b[i+1, j-1]-b[i,j-1])/x[i+j-1]-x[i]
    xt = 1
    yint = b[1,1]
    for j in range(1, n-1):
        xt = xt * (xx-x[j])
        yint = yint + b[1,j+1]*xt
    return yint
def Lagrange(x, y, xx):
    n = len(x)
    if (len(y) != n):
        print("ERROR : x and y must be same length")
        return
    s = 0
    for i in range(1,n):
        product = y[i]
        for j in range(1,n):
            if (i != j):
                product = product * (xx - x[j]) / (x[i] - x[j])
        s = s + product
    return s

x = np.array([1.0,4.0,6.0,5.0])
y = np.log(x)
print(Newint(x,y,2))

T = np.array([-40.0, 0.0, 20.0, 50.0])
d = np.array([1.52, 1.29, 1.2, 1.09])
density = Lagrange(T, d, 15)

print(density)