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

def bisect_2(func, xl, xu, es = 0.0001, maxit = 100):

    test = func(xl) * func(xu)

    if test > 0:
        print("No sign change")
        return [], [], [], []

    xr = xl
    n = round(np.log2((xu - xl)/es) + 0.5)

    for i in range(n):
        xroid = xr
        xr = (xl + xu)/2
        if xr != 0:
            ea = abs((xr - xroid)/xr) * 100

        ea = abs(xr - xroid)

        test = func(xl) * func(xr)
        if (test < 0):
            xu = xr
        elif test > 0:
            xl = xr
        else:
            ea = 0

    root = xr
    fx = func(xr)

    return root, fx, ea, n

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

def incsearch(func, xmin, xmax, ns):
    # x 범위 설정
    x = np.linspace(xmin, xmax, ns)
    # 주어진 함수 실행 배열 세팅
    f = func(x)
    # 구간값의 개수
    nb = 0
    # 구간값을 보관할 배열
    xb = []
    # 0부터 설정한 최고값까지 반복
    for k in np.arange(np.size(x) - 1):
        # f(x), f(x+1) 부호 다르면
        if np.sign(f[k]) != np.sign(f[k+1]):
            # 개수 추가
            nb += 1
            # 여기부터 여기까지
            xb.append(x[k])
            xb.append(x[k+1])
    # 구간 개수, 구간 배열
    return nb, xb










x = np.linspace(-1, 0, 20)
fp = (-12) - 21*x + 18*(x**2) - 2.75*(x**3)
plt.title("m graph")
plt.plot(x, fp)
plt.grid(color = 'b', linestyle = "--", linewidth = 1)
plt.show()

print("=========================")

print("[!] 2. Bisection")
fx = lambda x: (-12) - 21*x + 18*(x**2) - 2.75*(x**3)
root, fx, ea, iter = false_position(fx, -1, 0)
print("[+] root:", root)
print("[+] f(root):", fx, "(Must Be Zero)")
print("[+] Estimated Error:", ea, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter)

print("=========================")

print("[!] 3. false_position")
fx2 = lambda x: (-12) - 21*x + 18*(x**2) - 2.75*(x**3)
root2, fx2, ea2, iter2 = false_position(fx2, -1, 0)
print("[+] root2:", root2)
print("[+] f(root2):", fx2, "(Must Be Zero)")
print("[+] Estimated Error:", ea2, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter2)

print("=========================")

print("[!] 4. Bisection_2")
fc3 = lambda c: np.sqrt((9.81*80) / c) * np.tanh(np.sqrt(9.81 * c / 80) * 4) - 36
root3, fc3, ea3, n = bisect_2(fc3, 0.1, 0.2, 0.0001, 100)
print("[+] root3:", root3)
print("[+] f(root3):", fc3, "(Must Be Zero)")
print("[+] Estimated Error:", ea3, "(Must Be Zero Error)")
print("[+] n:", n)

