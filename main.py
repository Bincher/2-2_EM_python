import numpy as np
from numpy import linalg as LA

a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
b = np.array([[1,1/2,1/3],[1,2/3,1/2],[1,3/4,3/5]])

"""
print("a")
print(a)

# 스펙트랄(p=2) 놈에 근거한 조건수
print("\nLA.cond(a)")
print(LA.cond(a))

print("\nLA.cond(a, 'fro')")
print(LA.cond(a, 'fro'))

#행렬의 행-합 놈
print("\nLA.cond(a, np.inf)")
print(LA.cond(a, np.inf))

print("\nLA.cond(a, -np.inf)")
print(LA.cond(a, -np.inf))

print("\nLA.cond(a, 1)")
print(LA.cond(a, 1))

print("\nLA.cond(a, -1)")
print(LA.cond(a, -1))

print("\nLA.cond(a, 2)")
print(LA.cond(a, 2))

print("\nLA.cond(a, -2)")
print(LA.cond(a, -2))

print("\nmin(LA.svd(a, compute_uv=False))*min(LA.svd(LA.inv(a), compute_uv=False))")
print(min(LA.svd(a, compute_uv=False))*min(LA.svd(LA.inv(a), compute_uv=False)))
"""

print("b")
print(b)

# 스펙트랄(p=2) 놈에 근거한 조건수
print("\nLA.cond(b)")
print(LA.cond(b))

# 행렬의 행-합 놈
print("\nLA.norm(b, np.inf)")
print(LA.norm(b, np.inf))

# 역행렬의 햅-합 놈
print("\nLA.norm(LA.inv(b), np.inf)")
print(LA.norm(LA.inv(b), np.inf))

# 행-합 놈으로 구한 조건수
print("\nLA.cond(b, np.inf)")
print(LA.cond(b, np.inf))

t1 = LA.norm(b, np.inf)
t2 = LA.norm(LA.inv(b), np.inf)

print("\nLA.norm(LA.inv(b), np.inf) * LA.norm(b, np.inf)")
print(t1*t2)