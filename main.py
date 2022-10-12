import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve

def func(x):
    return (x*x - 9)

def Newton_Raphson(func, dfunc, xr, es = 0.0001, maxit = 50):
    iter = 0
    while(1):
        xrold = xr
        xr = xr - np.feval(func, xr)/np.feval(dfunc, xr)
        iter += 1
        if xr != 0:
            ea = abs((xr - xrold)/xr) * 100
        if ea <= es or iter >= maxit:
            break
    root = xr

    return root, ea, iter


x = fsolve(func,-4)
print(x)
x = fsolve(func, 4)
print(x)
x = fsolve(func, 0)
print(x)

"""
print("[!] 5. Newton_Raphson")
fx = lambda x: (2.78)**(-x) - x
fx2 = lambda x: (-2.78)**(-x) - 1
root, ea, iter = Newton_Raphson(fx, fx2, 0)
print("[+] root:", root)
print("[+] Estimated Error:", ea, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter)
"""