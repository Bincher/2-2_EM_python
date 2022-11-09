import numpy as np
from numpy import linalg as LA

"""
A = np.array([[10.0, 2.0, -1.0], [-3.0, -6.0, 2.0], [1.0, 1.0, 5.0]])
b = np.array([[27.0],[-61.5],[-21.5]])

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

"""
A = np.array([[10.0, 2.0, -1.0], [-3.0, -6.0, 2.0], [1.0, 1.0, 5.0]])
b = np.array([[27.0],[-61.5],[-21.5]])

Ad = LA.linalg.inv(A)
print("역행렬")
print(Ad)
print("[A][A]^-1")
print(LA.linalg.dot(A,Ad))
"""

"""
A = np.array([[-8.0, 1.0, -2.0], [-2.0, -6.0, -1.0], [-3.0, -1.0, 7.0]])
b = np.array([[-20.0],[-38.0],[-34.0]])

Ad = LA.linalg.inv(A)
print("역행렬")
print(Ad)
"""

"""
a = np.array([[8.0, 2.0, -10.0], [-9.0, 1.0, 3.0], [15.0, -1.0, 6.0]])
b = np.array([[8.0/-10.0, 2.0/-10.0, -10.0/-10.0], [-9.0/-9.0, 1.0/-9.0, 3.0/-9.0], [15.0/15.0, -1.0/15.0, 6.0/15.0]])

print("a")
print(a)

print("b")
print(b)

print("Af")
print(LA.norm(b,'fro'))

print("At")
print(LA.norm(b, 1))

print("Ainf")
print(LA.norm(b, np.inf))

# 스펙트랄(p=2) 놈에 근거한 조건수
print("\nLA.cond(b) 스펙트랄(p=2) 놈에 근거한 조건수")
print(LA.cond(b))

# 행렬의 행-합 놈
print("\nLA.norm(b, np.inf) 행렬의 행-합 놈")
print(LA.norm(b, np.inf))

# 역행렬의 햅-합 놈
print("\nLA.norm(LA.inv(b), np.inf) 역행렬의 햅-합 놈")
print(LA.norm(LA.inv(b), np.inf))

# 행-합 놈으로 구한 조건수
print("\nLA.cond(b, np.inf) 행-합 놈으로 구한 조건수")
print(LA.cond(b, np.inf))

t1 = LA.norm(b, np.inf)
t2 = LA.norm(LA.inv(b), np.inf)

print("\nLA.norm(LA.inv(b), np.inf) * LA.norm(b, np.inf)")
print(t1*t2)
"""

"""
A = np.array([[-8.0, 1.0, -2.0], [-2.0, -6.0, -1.0], [-3.0, -1.0, 7.0]])
B = np.array([[15.0, -3.0, -1.0], [-3.0, 18.0, -6.0], [-4.0, -1.0, 12.0]])

print("Af")
print(LA.norm(A,'fro'))

print("Ainf")
print(LA.norm(A, np.inf))

print("Bf")
print(LA.norm(B,'fro'))

print("Binf")
print(LA.norm(B, np.inf))
"""

x1 = 4.0
x2 = 2.0
x3 = 7.0
A = np.array([[x1**2, x1, 1.0], [x2**2, x2, 1.0], [x3**2, x3, 1.0]])

print("행-합 놈에 근거한 조건수 계산")
print(LA.cond(A, np.inf))

print("\n스펙트랄 놈에 근거한 조건수")
print(LA.cond(A))
print("LA.cond(A, 'fro')")
print(LA.cond(A,'fro'))