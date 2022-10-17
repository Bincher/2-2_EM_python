import numpy as np

K = np.array([[150, -100, 0], [-100, 150, -50], [0, -50, 50]])
mg = np.array([588.6, 686.7, 784.8])

x1 = np.linalg.inv(K) * mg
print(x1)

x2 = np.dot(np.linalg.inv(K), mg)
print(x2)

x = np.linalg.inv(K)

print(x)

xi = np.array([20, 40, 60])
xf = x + xi
print(xf)