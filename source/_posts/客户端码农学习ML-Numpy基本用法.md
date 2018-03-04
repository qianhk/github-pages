---
title: 客户端码农学习ML —— Numpy基本用法
date: 2018-03-02 22:12:43
tags: [Numpy]
articleID: 客户端码农学习ML-Numpy基本用法
---

本文从创建矩阵、维度变换、矩阵运算、随机数、索引等方面总结numpy中基本用法，脚本首先import numpy as np。

## 创建矩阵

```
np.array([1, 2, 3])

输出 [1 2 3]
```

```
np.array([(1, 2, 3), (4, 5, 6)], dtype=np.int32)

指定类型int32，输出
[[1 2 3]
 [4 5 6]] 
```

<!-- more -->

```
np.zeros((2, 3))
np.ones((2, 3), dtype=int)
np.eye(3)

创建全为0、全为1的矩阵，以及单位矩阵
```

```
np.arange(12)

输出[0, 12)，[ 0  1  2  3  4  5  6  7  8  9 10 11]

np.arange(1, 2, 0.3)

1开头 差值为0.3的等差数列，直到小于2
输出 [ 1.   1.3  1.6  1.9]
```

```
np.linspace(1, 2, 11)

1开头 2最后，中间11的数字平分,输出：[ 1.   1.1  1.2  1.3  1.4  1.5  1.6  1.7  1.8  1.9  2. ]
```

> 首先定义a、b两个矩阵
> 
```
A = np.array([[1, 2, 3]
            , [4, 5, 6]])
B = np.array([[2, 1, 0]
            , [1, 1, 1]])
```
>

## 维度变换

shape重定义

```
A = np.arange(1, 7).reshape(2, 3)

上述矩阵a还可以通过reshape改变1维数组为2行3列
[[1 2 3]
 [4 5 6]]
```

矩阵转置

```
C = A.T
C = A.transpose()

原先2行3列的矩阵变成了3行2列
[[1 4]
 [2 5]
 [3 6]]
```

逆矩阵(必须是方阵，即行、列的维度相等，同时det(D) != 0)

```
D = np.mat("1 2 3; 2 3 1; 3 1 2")

D.I

[[-0.27777778  0.05555556  0.38888889]
 [ 0.05555556  0.38888889 -0.27777778]
 [ 0.38888889 -0.27777778  0.05555556]]

np.dot(D, D.I) = np.dot(D.I, D) = I 
```

水平组合: 要求横轴即1维上的数量相同

```
np.hstack((A, B))

 [[1 2 3 2 1 0]
 [4 5 6 1 1 1]]
```

垂直组合: 要求纵轴即0维上的数量相同

```
np.vstack((A, B))

 [[1 2 3]
 [4 5 6]
 [2 1 0]
 [1 1 1]]

```

水平拆分

```
np.hsplit(A, 3)

[array([[1],[4]]), 
 array([[2],[5]]),
 array([[3],[6]])]
```

垂直拆分

```
np.vsplit(A, 2)

[array([[1, 2, 3]]), array([[4, 5, 6]])]
```

## 矩阵运算

矩阵与标量的加减乘除等于矩阵内各元素与标量的加减乘除

```
A + 1

[[2 3 4]
 [5 6 7]]

A * 2
[[ 2  4  6]
 [ 8 10 12]]

```

矩阵与矩阵相加，各维度必须一致，相同位置的元素相加，否则报错

```
A + B

[[3 3 3]
 [5 6 7]]
```

用dot方法相乘，表示线性代数里的向量内积(也叫点积、数量积)或矩阵乘法

要注意向量内积满足交换律但不满足结合律，相乘的结果是一个数。

而矩阵乘法满足结合律但不满足交换律，相乘的结果还是一个矩阵。

```
np.dot(A, C)

[[14 32]
 [32 77]]
 
但是当秩为1时，也是对应位置元素相乘并累加。
```

用multiply相乘，维度也必须一致，相同位置的元素相乘

```
np.multiply(A, B)

[[2 2 0]
 [4 5 6]]
```

A, B创建后的类型是type(A) = 'numpy.ndarray', 如果显式转换成matrix类型，那么*跟dot含义一致，否则跟multiply一致。通过np.mat(A)转换为'numpy.matrixlib.defmatrix.matrix'。

```
A * B

[[2 2 0]
 [4 5 6]]
```

```
np.mat(A) * np.mat(C)

[[14 32]
 [32 77]]
```

用向量试试：

```
M = np.array([1, 2, 3])
N = np.array([1, 0, 2])
print('\nM = %s    N = %s type(M)=%s' % (M, N, type(M)))

DOT_MN = np.dot(M, N)
print('dot(M,N)=%s  type=%s' % (DOT_MN, type(DOT_MN)))

DOT_MAT_MN = np.dot(np.mat(M), np.mat(N).T)
print('dot(mat(M), mat(N).T)=%s  type=%s' % (DOT_MAT_MN, type(DOT_MAT_MN)))

输出如下：
M = [1 2 3]    N = [1 0 2] type(M)=<class 'numpy.ndarray'>
dot(M,N)=7  type=<class 'numpy.int64'>
dot(mat(M), mat(N).T)=[[7]]  type=<class 'numpy.matrixlib.defmatrix.matrix'>
```

总结下来，即:

multiply始终是数乘，相同位置元素相乘

dot始终是向量内积或者矩阵乘法

而*根据数据类型决定如何乘

## 随机数

```
np.random.random((2, 3))

生成[0,1)之间的浮点数
[[ 0.96702984  0.54723225  0.97268436]
 [ 0.71481599  0.69772882  0.2160895 ]]
 
从python源码的注释看，ranf = random = sample = random_sample
```

```
np.random.rand(2, 4)

[[ 0.97627445  0.00623026  0.25298236  0.43479153]
 [ 0.77938292  0.19768507  0.86299324  0.98340068]]

同样生成生成[0,1)之间的浮点数，与random的具体区别不是很清楚，从注释上看rand属于uniform distribution，random属于continuous uniform
```

```
np.random.randn(2, 4)

生成标准正态分布样本 0为均值、1为标准差 N(0,1)
[[ 0.33225003 -1.14747663  0.61866969]
 [-0.08798693  0.4250724   0.33225315]]
```

```
np.random.normal(10, 1, (2, 6))

生成均值为loc，标准差为scale的正态分布矩阵，numpy中很多方法如果size不写则返回一个值
[[  9.43149964   9.05681477  10.55712148   9.97022176  10.41476467]
 [ 10.35518302  10.35732679   9.05575841  12.32206439  10.31706671]]

```

```
np.random.randint(10, 20, size=(2, 5))

生成[10, 20)之间的随机整数
[[11 10 13 12 13]
 [12 19 12 11 10]]
```

```
np.random.choice(10, 5, False)

从[0, 10)选5个不重复的数: [9 6 8 7 5]

list = ['a', 'b', 'c', 'd', 'e']
np.random.choice(list, size=(3, 4), replace=True)

从另一个数组中选择可重复值
[['c' 'b' 'c' 'd']
 ['a' 'e' 'd' 'd']]

```

## 索引与切片

对于一维数组

x = [0 1 2 3 4 5 6 7 8 9]

```
x[2] : 1
x[2:5] : [2 3 4]

甚至还可以用数组当索引
indexs = [2, 3, 4]
(x * 2)[indexs]

输出[4, 6, 8]

x[::-1]
x[::-2]
如果第二个冒号后是负数，则数组反转，非-1则类似正数间隔筛选:
x[::-1] = [9 8 7 6 5 4 3 2 1 0]
x[::-2] = [9 7 5 3 1]
```

对于多维矩阵，每个维度之间用,分隔开，单独用类似于一维数组的方式指定索引

```
print('A[1] = %s\n' % A[1])
print('A[1, :] = %s\n' % A[1, :])
print('A[:, 1] = %s\n' % A[:, 1])
print('A[1:3, 1:3] = %s\n' % A[1:3, 1:3])

A[1] = [4 5 6]
A[1, :] = [4 5 6]
A[:, 1] = [2 5]
A[1:3, 1:3] = [[5 6]]
```

## Boolean Arrays

```
X = np.arange(12).reshape(3,4)
X = X > 4

[[False False False False]
 [False  True  True  True]
 [ True  True  True  True]]
 
 
X.any() : 是否有True? True
X.all() : 是否都是True? False
```

## 矩阵自身属性

A.ndim A.shape A.size A.dtype A.itemsize

分别表示：矩阵维数、各维度具体大小、元素总数、元素类型、此类型占用字节数

输出分别是：2  (2, 3) 6 int64 8

## 其它操作

求和、最大、最小值、平均值、方差、标准差

```
A.sum(), A.min(), A.max(), A.mean(), A.var(), A.std()

sum=21 min=1 max=6 mean=3.5 var=2.9167 std=1.7078

```

提取矩阵对角: 如果原来是1维矩阵则转为对角矩阵，否则提取对角线返回1维矩阵


```
np.diag(A)

[1 5]

np.diag(np.diag(A))

[[1 0]
 [0 5]]

```

## 总结

很强大的矩阵操作库，其它还有大量功能待后续继续了解，如concatenate、dstack、column_stack、split、sina、sqrt、cumsum、fromfunction、floor、resize、where等等，还可以通过save、load、savetxt、loadtxt等进行文件读写。

## 参考

https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

http://blog.csdn.net/zenghaitao0128/article/details/78715140

http://codingpy.com/article/an-introduction-to-numpy/

## 　

本文首发于[钱凯凯的博客](http://qianhk.com) : http://qianhk.com/2018/03/客户端码农学习ML-Numpy基本用法/
