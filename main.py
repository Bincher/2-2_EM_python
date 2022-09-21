import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""
#축 라벨, 제목, 범례 사용(sin,cos 2개 겹치기)
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel("x value")
plt.ylabel("y value")
plt.title("sine & cosine")
plt.legend(["sine", "cosine"])
plt.show()

#복수개의 라인을 한개로(점)
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

#sin과 cos 그래프 2개
X = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(X)
y_cos = np.cos(X)
plt.subplot(2, 1, 1)
plt.plot(X, y_sin)
plt.title('sine')
plt.subplot(2, 1, 2)
plt.plot(X, y_cos)
plt.title('cosine')
plt.show()

#격자, 선 속성, 격자 간격
X = np.linspace(-1.5, 1.5, 100)
F1 = X ** 2
F2 = X ** 3
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
plt.plot(X, F1, color='m', linestyle='dotted', linewidth=2.)
plt.plot(X, F2, c='#0000FF', ls='dashed', lw=1.)
plt.grid(True, which='both', c='0.5', ls='--', lw=0.5)
plt.show()
"""
