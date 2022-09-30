import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#xl : 계산할 x의 최저값
#xu : 계산할 x의 최고값
#ex :오차 범위

def bisect(func, xl, xu, es = 2, maxit = 100):
    # f(최저값)과 x(최고값)을 곱함
    test = func(xl) * func(xu)

    if (test > 0):
        print("No sign change")
        return [], [], [], []

    iter = 0

    xr = xl
    ea = 100

    while True:
        xroid = xr
        xr = float((xl + xu) / 2)
        iter += 1

        print("iter: ",iter)
        print("xr: ",xr)

        if xr != 0:
            ea = np.abs((xr-xroid)/xr) * 100

        test = func(xl) * func(xr)

        if test < 0:
            xu = xr
        elif test > 0:
            xl = xr
        else:
            ea = 0

        if (ea < es or iter >= maxit):
            break;

    root = xr

    fx = func(xr)

    return root, fx, ea, iter

def false_position(func, xl, xu, es = 1.0e-4, maxit = 100):
    # f(최저값)과 x(최고값)을 곱함
    test = func(xl) * func(xu)

    if (test > 0):
        print("No sign change")
        return [], [], [], []

    iter = 0

    xr = xl
    ea = 100

    while True:
        xroid = xr

        xr = xu - (func(xu) * (xl - xu)) / (func(xl) - func(xu))
        iter += 1

        print("iter: ",iter)
        print("xr: ",xr)

        if xr != 0:
            ea = np.abs((xr-xroid)/xr) * 100

        test = func(xl) * func(xr)

        if test < 0:
            xu = xr
        elif test > 0:
            xl = xr
        else:
            ea = 0

        if (ea <= es or iter >= maxit):
            break;

    root = xr
    fx = func(xr)

    return root, fx, ea, iter

print("[!] 5.1) 번지점프, 이분법, 근사 상대오차가 2%이하로")
fc = lambda c: np.sqrt((9.81*80) / c) * np.tanh(np.sqrt(9.81 * c / 80) * 4) - 36
root, fx, ea, iter = bisect(fc, 0.1, 0.2)
print("[+] root:", root)
print("[+] f(root):", fx, "(Must Be Zero)")
print("[+] Estimated Error:", ea, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter)

"""
print("[!] 2. Bisection\n")
fm = lambda m: np.sqrt(9.81 * m / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / m) * 4) - 36
root, fx, ea, iter = bisect(fm, 40, 200, 0.001, 50)
print("[+] root:", root)
print("[+] f(root):", fx, "(Must Be Zero)")
print("[+] Estimated Error:", ea, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter)

print("=========================")
print("[!] 3. false_position\n")
fm2 = lambda n: pow(n, 10) - 1
root2, fx2, ea2, iter2 = false_position(fm2, 0, 1.3, 0.35, 100)
print("[+] root:", root2)
print("[+] f(root):", fx2, "(Must Be Zero)")
print("[+] Estimated Error:", ea2, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter2)
"""

