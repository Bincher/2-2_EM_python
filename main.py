import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve

def Newton_Raphson(func, dfunc, xr, es = 5, maxit = 300):

    iter = 0

    while(1):
        xrold = xr
        xr = xr - func(xr)/dfunc(xr)
        iter += 1

        if xr != 0:
            ea = (abs((xr - xrold)/xr) * 100)

        if ea <= es or iter >= maxit:
            break
    root = xr

    return root, ea, iter

x = np.linspace(-5, 5, 20)
fp = -2*pow(x,6) - 1.5*pow(x,4) + 10*x + 2
plt.title("m graph")
plt.plot(x, fp)
plt.grid(color = 'b', linestyle = "--", linewidth = 1)
plt.show()
#fx = lambda x: x**3 - 6*(x**2) + 11*x - 6.1
#fx2 = lambda x: 3*(x**2) - 12*x + 11
#fx = lambda x: -2*pow(x,6) - 1.5*pow(x,4) + 10*x + 2
#fx2 = lambda x: -12*pow(x,5) - 6*pow(x,3) + 10

print("[!] 5. Newton_Raphson")
fx = lambda x: -12*pow(x,5) - 6*pow(x,3) + 10
fx2 = lambda x: -60*pow(x,4) - 18*pow(x,2)
root, ea, iter = Newton_Raphson(fx, fx2, 1)
print("[+] root:", root)
print("[+] Estimated Error:", format(ea, '.5f'), "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter)
print("극점이 최대값일테니... :",-2*pow(root,6) - 1.5*pow(root,4) + 10*root + 2)