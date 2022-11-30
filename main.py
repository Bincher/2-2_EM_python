import numpy as np
import matplotlib.pyplot as plt

#외삽법의 위험성
"""
t = np.array([i for i in range(1920,1991,10)])
print("x축(year) : ", t)

pop = np.array([106.46, 123.08, 132.12, 152.27, 180.67, 205.05, 227.23, 249.46])
p = np.polyfit(t, pop, 7)

ts = (t-1955)/35
print(ts)

p = np.polyfit(ts,pop,7)
print(p)

q = np.polyval(p,(2000-1955)/35)
print(q)

tt = np.linspace(1920, 2000)
pp = np.polyval(p, (tt-1955)/35)
plt.plot(t, pop, 'o', tt, pp)
plt.show()
"""

#고차 다항식보간법의 위험성
"""
x = np.linspace(-1, 1, 5)
y = 1. / (1 + 25*x**2)
xx = np.linspace(-1,1)

p = np.polyfit(x, y, 4)
y4 = np.polyval(p, xx)

yr = 1. / (1 + 25*xx**2)
plt.plot(x, y, 'o', xx, y4, xx, yr, '--')
plt.show()

x = np.linspace(-1, 1, 11)
y = 1 / (1+25*x**2)
p = np.polyfit(x,y,10)
y10 = np.polyval(p, xx)
plt.plot(x, y, 'o', xx, y10, xx, yr, '--')
plt.show()
"""
