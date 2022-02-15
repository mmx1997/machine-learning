import numpy as np

seed = 1
np.random.seed(seed)

# array 方法
a = np.array([1, 2, 3])
b = np.array([1.2, 2.3, 3.4])
print('a: ', a)  # a:  [1 2 3]
print('type of a: ', a.dtype)  # type of a:  int64
print('type of d: ', b.dtype)  # type of d:  float64

# 转换多维数组
print('-' * 20, '转换多维数组', '-' * 20)
c = np.array([[1, 2, 3], [4, 5, 6]])
print(c)
# [[1 2 3]
#  [4 5 6]]

# 其他生成数组的方法
print('-' * 20, '其他生成数组的方法', '-' * 20)
print('np.zeros((2, 3)): \n', np.zeros((2, 3)))
# [[0. 0. 0.]
#  [0. 0. 0.]]
print('np.ones((3, 4), dtype=np.int): \n', np.ones((3, 4), dtype=np.int))
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
print('np.empty((2, 3)): \n', np.empty((2, 3)))
# [[0. 0. 0.]
#  [0. 0. 0.]]
print('np.arange(10,30,5): \n', np.arange(10, 30, 5))
# [10 15 20 25]


# Numpy random 提供的生成随机数组的方法
print('-' * 10, 'Numpy random 提供的生成随机数组的方法', '-' * 10)
print('均匀分布 np.random.rand(3, 2): \n', np.random.rand(3, 2))
# [[4.17022005e-01 7.20324493e-01]
#  [1.14374817e-04 3.02332573e-01]
#  [1.46755891e-01 9.23385948e-02]]
print('np.random.randint(3, size=5): \n', np.random.randint(3, size=5))
# [0 1 0 2 1]
print('np.random.randint(3, size=(2, 3)): \n', np.random.randint(3, size=(2, 3)))
# [[2 0 2]
#  [1 2 0]]
print('标准正态分布 np.random.randn(3): \n', np.random.randn(3))
# [0.80074457 0.69232262]

# 数组的索引与切片
print('-' * 10, '数组的索引与切片', '-' * 10)
a = np.arange(10) ** 2
print(a)  # [ 0  1  4  9 16 25 36 49 64 81]

print('第二个元素：', a[2])  # 4
print('第2到4个元素：', a[1:4])  # [1 4 9]
print('翻转数组：', a[::-1])  # [81 64 49 36 25 16  9  4  1  0]

b = np.random.random((3, 3))
print(b)
# [[0.14038694 0.19810149 0.80074457]
#  [0.96826158 0.31342418 0.69232262]
#  [0.87638915 0.89460666 0.08504421]]

print('第1行第2列元素：', b[1, 2])  # 0.6923226156693141
print('第1列元素', b[:, 1])  # [0.19810149 0.31342418 0.89460666]
print('前两行第2列元素：', b[:2, 2])  # [0.80074457 0.69232262]

# 数组的基础运算
print('-' * 20, '数组的基础运算', '-' * 20)

a = np.arange(4)
b = np.array([5, 10, 15, 20])

print('b - a: ', b - a)  # [ 5  9 13 17]
print('b ** 2: ', b ** 2)  # [ 25 100 225 400]
print('np.sin(a): ', np.sin(a))  # [0.         0.84147098 0.90929743 0.14112001]
print('b < 20: ', b < 20)  # [ True  True  True False]
print('np.mean(b): ', np.mean(b))  # 12.5
print('np.var(b): ', np.var(b))  # 31.25

print('-' * 20, '数组的线性代数运算', '-' * 20)
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])

print('A * B: ', A * B)  # [[2 0] [0 4]]
print('A.dot(B): ', A.dot(B))  # [[5 4] [3 4]]
print('np.linalg.inv(A): ', np.linalg.inv(A))  # [[ 1. -1.] [ 0.  1.]]
print('np.linalg.det(A): ', np.linalg.det(A))  # 1.0

print('-' * 20, '数组维度变换', '-' * 20)
a = np.floor(10 * np.random.random((3, 4)))
print('a: \n', a)
'''
 [[0. 1. 8. 0.]
 [4. 9. 5. 6.]
 [3. 6. 8. 0.]]
'''
print('a.shape: ', a.shape)  # (3, 4)
print('a.ravel(): ', a.ravel())  # [0. 1. 8. 0. 4. 9. 5. 6. 3. 6. 8. 0.]
print('a.reshape(2,6): \n', a.reshape(2, 6))
'''
[[0. 1. 8. 0. 4. 9.]
 [5. 6. 3. 6. 8. 0.]]
'''
print('a.T: \n', a.T)
'''
 [[0. 4. 3.]
 [1. 9. 6.]
 [8. 5. 8.]
 [0. 6. 0.]]
'''
print('a.T.shape: ', a.T.shape)  # (4, 3)
print('a.reshape(3, -1): \n', a.reshape(3, -1))
'''
 [[0. 1. 8. 0.]
 [4. 9. 5. 6.]
 [3. 6. 8. 0.]]
'''

print('-' * 20, '数组的合并与切分', '-' * 20)

print('np.hstack(A, B): \n', np.hstack((A, B)))
'''
[[1 1 2 0]
 [0 1 3 4]]
'''
print('np.vstack(A, B): \n', np.vstack((A, B)))
'''
 [[1 1]
 [0 1]
 [2 0]
 [3 4]]
'''

C = np.arange(16.).reshape(4, 4)
print('np.arange(16.).reshape(4, 4) C: \n', C)
'''
 [[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]
 [12. 13. 14. 15.]]
'''

print('np.hsplit(C, 2): \n', np.hsplit(C, 2))
'''
[array([[ 0.,  1.],
       [ 4.,  5.],
       [ 8.,  9.],
       [12., 13.]]), array([[ 2.,  3.],
       [ 6.,  7.],
       [10., 11.],
       [14., 15.]])]
'''
print('np.vsplit(C, 2): \n', np.vsplit(C, 2))
'''
 [array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]]), array([[ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]])]
'''
