import matplotlib
import numpy as np
import matplotlib.pyplot as pit
from numpy import linalg as LA

A = np.array([[90000, 300, 1], [160000, 400, 1], [250000, 500, 1]])
print(LA.cond(A))

T = np.array([300, 400, 500])
density = np.array([0.616, 0.525, 0.457])
p = np.polyfit(T, density, 2)

print(p)

#350도에서 밀도
d = np.polyval(p, 350)
print(d)