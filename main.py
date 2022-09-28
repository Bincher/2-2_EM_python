import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""
cd = 0.25
g = 9.81
v = 36
t = 4
mp = np.linspace(50, 200, 10)
fp = np.sqrt(g*mp/cd) * np.tanh(np.sqrt(g*cd/mp)*t) - v

plt.title("m graph")
plt.plot(mp, fp)
plt.grid(color = 'b', linestyle = "--", linewidth = 1)
plt.show()



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

print("[!] 1. Incremental Search\n")

fx = lambda x: np.sin(10+x) + np.cos(3*x)
nb, xb = incsearch(fx, 3, 6, 50)

# 구간 갯수
print("[+] Number of Brackets =", nb)
# 구간들
print("[+] Root Interval =", xb)
"""

def bisect(func, xl, xu, es = 1.0e-4, maxit = 100):
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

        if (ea <= es or iter >= maxit):
            break;

    root = xr
    fx = func(xr)

    return root, fx, ea, iter

print("[!] 2. Bisection\n")
fm = lambda m: np.sqrt(9.81 * m / 0.25) * np.tanh(np.sqrt(9.81 * 0.25 / m) * 4) - 36
root, fx, ea, iter = bisect(fm, 40, 200, 0.001, 50)
print("[+] root:", root)
print("[+] f(root):", fx, "(Must Be Zero)")
print("[+] Estimated Error:", ea, "(Must Be Zero Error)")
print("[+] Iterated Number to Find Root:", iter)