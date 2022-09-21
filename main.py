import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt

a = np.array([1,2,3,4])
print("a", a)
print("a + 1", a + 1)
print("2**a",2**a)
print("==============")

b = np.ones(4) + 1 #ones 모든 원소가 1인 배열 생성(크기는 4)
print("b", b)
print("a-b", a-b)
print("a*b", a*b)
print("==============")

j = np.arange(5) #01234
print("2**(j + 1)-j", 2**(j + 1)-j)

c = np.ones((3,3))
print("c\n", c)
print("c*c\n", c * c) #행렬 스칼라곱
print("c.dot(c)\n", c.dot(c)) #행렬 내적(선형대수)
print("a==b", a==b)
print("np.logical_and(a,b)", np.logical_and(a,b)) #dtype = bool
print("==============")

d = np.arange(5)
print("d", d)
print("np.sin(d)", np.sin(d))
print("np.log(d)", np.log(d))
print("np.exp(d)", np.exp(d))
print("==============")

e = np.triu(np.ones((3,3)), 1) # 0 1 1
print("e\n", e)                # 0 0 1
print("e.T\n", e.T) # 반대로    # 0 0 0
e.T[0,2] = 999
print("e.T[0.2]\n",e.T)
print("e\n", e)
print("==============")

print("a", a)
print("np.sum(a)", np.sum(a)) #합
print("a.sum()", a.sum()) #합
print("==============")

f = np.array([[1,1], [2,2]])
print("f\n", f)
print("f.sum(axis=0)", f.sum(axis=0)) # [3 3]
print("f[:, 0].sum(), f[:, 1].sum()\n",f[:, 0].sum(), f[:, 1].sum()) # 3 3
print("f.sum(axis=1)", f.sum(axis=1)) # [2 4]
print("f[0, :].sum(), f[1, :].sum()\n",f[0, :].sum(), f[1, :].sum()) # 2 4
print("==============")

g = np.array([1,3,2])
print("g\n", g)
print("g.min()", g.min())
print("g.max()", g.max())
print("g.argmin()", g.argmin()) #최솟값의 index
print("g.argmax()", g.argmax()) #최댓값의 index
print("==============")

h = np.array([1,2,3,1])
i = np.array([[1,2,3],[5,6,1]])
print("h.mean()",h.mean()) #평균
print("np.median(h)",np.median(h)) #중앙값
print("np.median(i, axis=-1)",np.median(i, axis=-1)) # [2. 5. ]
print("h.std()",h.std()) #표준편차
print("==============")

#두 배열 형태가 다른 경우 브로드캐스팅 기법 사용
arr1 = np.array([[[3, 5, 8]]])
arr2 = np.array([[[3], [5], [8]]])
arr3 = np.array([[[3]], [[5]], [[8]]])
print("arr1.shape",arr1.shape)
print("arr2.shape",arr2.shape)
print("arr3.shape",arr3.shape)
print("==============")

a1 = np.array([[1, 2], [3,4]])
print("a1\n",a1)
a2 = a1.flatten()
a1[0][0] = 15
print("a1.flatten()\n",a2)
a3 = a1.ravel() # a1.reshape(-1)
a1[0][0] = 15
print("a1.ravel()\n",a3)
print("==============")

j = np.array([[4,3,5],[1,2,1]])
k = np.sort(j, axis = 1)
print("j\n",j)
print("np.sort(j, axis = 1)\n",k)
j.sort(axis = 1)
print("j.sort(axis = 1)\n",j)
print("==============")

l = np.array([4,3,1,2])
m = np.argsort(l)
print("l", l)
print("np.argsort(l)",m)
print("l[np.argsort(l)]",l[m])
print("np.argmax(l)",np.argmax(l))
print("np.argmin(l)",np.argmin(l))
print("==============")

n = np.linspace(0,10,11)
#시작값 = 0, 끝값 = 10, 간격개수 = 11(default = 50), endpoint=True, retstep=True)
num = len(n)
o1=(2*n + 1.5)
o2=np.zeros(num)
print("n", n)
print("o1=(2*n + 1.5)", o1)
print("o2", o2)
rd.seed()
for i in range(0, num):
    noise=rd.uniform(-3,3) #-3~3 사이의 난수
    print("noise = rd.uniform(-3,3)", i , "=" , noise)
    o2[i]=o1[i]+noise
plt.scatter(n, o1, c='black', s=80)
plt.scatter(n, o2, c='red', s=80)
plt.show()
print(n)
print(o2)
print("==============")

p = np.arange(0, 3*np.pi, 0.1)
q = np.sin(p)
print("sin : plt.plot(p, q)", plt.plot(p, q))
q = np.cos(p)
print("cos : plt.plot(p, q)", plt.plot(p, q))
q = np.tan(p)
print("tan : plt.plot(p, q)", plt.plot(p, q))
plt.show()
print("==============")

"""
a [1 2 3 4]
a + 1 [2 3 4 5]
2**a [ 2  4  8 16]
==============
b [2. 2. 2. 2.]
a-b [-1.  0.  1.  2.]
a*b [2. 4. 6. 8.]
==============
2**(j + 1)-j [ 2  3  6 13 28]
c
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
c*c
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
c.dot(c)
 [[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
a==b [False  True False False]
np.logical_and(a,b) [ True  True  True  True]
==============
d [0 1 2 3 4]
np.sin(d) [ 0.          0.84147098  0.90929743  0.14112001 -0.7568025 ]
np.log(d) [      -inf 0.         0.69314718 1.09861229 1.38629436]
np.exp(d) [ 1.          2.71828183  7.3890561  20.08553692 54.59815003]
==============
e
 [[0. 1. 1.]
 [0. 0. 1.]
 [0. 0. 0.]]
e.T
 [[0. 0. 0.]
 [1. 0. 0.]
 [1. 1. 0.]]
e.T[0.2]
 [[  0.   0. 999.]
 [  1.   0.   0.]
 [  1.   1.   0.]]
e
 [[  0.   1.   1.]
 [  0.   0.   1.]
 [999.   0.   0.]]
==============
a [1 2 3 4]
np.sum(a) 10
a.sum() 10
==============
f
 [[1 1]
 [2 2]]
f.sum(axis=0) [3 3]
f[:, 0].sum(), f[:, 1].sum()
 3 3
f.sum(axis=1) [2 4]
f[0, :].sum(), f[1, :].sum()
 2 4
==============
g
 [1 3 2]
g.min() 1
g.max() 3
g.argmin() 0
g.argmax() 1
==============
h.mean() 1.75
np.median(h) 1.5
np.median(i, axis=-1) [2. 5.]
h.std() 0.82915619758885
==============
arr1.shape (1, 1, 3)
arr2.shape (1, 3, 1)
arr3.shape (3, 1, 1)
==============
a1
 [[1 2]
 [3 4]]
a1.flatten()
 [1 2 3 4]
a1.ravel()
 [15  2  3  4]
==============
j
 [[4 3 5]
 [1 2 1]]
np.sort(j, axis = 1)
 [[3 4 5]
 [1 1 2]]
j.sort(axis = 1)
 [[3 4 5]
 [1 1 2]]
==============
l [4 3 1 2]
np.argsort(l) [2 3 1 0]
l[np.argsort(l)] [1 2 3 4]
np.argmax(l) 0
np.argmin(l) 2
==============
n [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
o1=(2*n + 1.5) [ 1.5  3.5  5.5  7.5  9.5 11.5 13.5 15.5 17.5 19.5 21.5]
o2 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
noise = rd.uniform(-3,3) 0 = -2.7876445372780396
noise = rd.uniform(-3,3) 1 = -0.8677459524173754
noise = rd.uniform(-3,3) 2 = 0.1570959462105459
noise = rd.uniform(-3,3) 3 = -0.6980732801285217
noise = rd.uniform(-3,3) 4 = 0.06779167766558425
noise = rd.uniform(-3,3) 5 = 0.6385960112313223
noise = rd.uniform(-3,3) 6 = -0.6216771439009063
noise = rd.uniform(-3,3) 7 = -1.2668669114536941
noise = rd.uniform(-3,3) 8 = -1.4702196145320852
noise = rd.uniform(-3,3) 9 = 1.808938747685822
noise = rd.uniform(-3,3) 10 = 1.0248806420880934
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
[-1.28764454  2.63225405  5.65709595  6.80192672  9.56779168 12.13859601
 12.87832286 14.23313309 16.02978039 21.30893875 22.52488064]
==============
sin : plt.plot(p, q) [<matplotlib.lines.Line2D object at 0x000002551537FD60>]
cos : plt.plot(p, q) [<matplotlib.lines.Line2D object at 0x00000255153C0070>]
tan : plt.plot(p, q) [<matplotlib.lines.Line2D object at 0x00000255153C0310>]
==============

Process finished with exit code 0

"""