# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 07:51:12 2018

@author HuangLingZhi
Email : hlzcow1986@126.com
# : 欢迎转载:转载请注明出处
Just used for numpy study
"""
# Just the numpy study
import numpy as np
# 参考：http://www.runoob.com/numpy/numpy-tutorial.html
# =============================================================================
# 1.基本用法
# =============================================================================

# creat a array
a1 = np.array([[1, 2, 3], [4, 5, 6]])
# 常用命令
# np.zeros((3,4), dtype=float) #  3行4列全为0的矩阵，数据类型为float
# np.ones((3,4), dtype=int) #  3行4列全为1的矩阵，数据类型为int
# np.full((3,4), 9, dtype=complex) # 3行4列的矩阵，填充为9, 数据类型为复数
# np.empty((3,4))  # 3行4列的空矩阵
# np.ones_like(a1)  # 返回一个和a1维数一致的矩阵,填充为1
# np.zeros_like(a1)  # 返回一个和a1维数一致的矩阵,填充为0
# np.empty_like(a1) # 和上例类似
# np.random.randint(0,10,(3,4))  # 返回一个3*4的矩阵，矩阵内容为0到10之间的随机整数
# np.random.randn(2,4)  # 返回一个2*4的矩阵，内容为符合(0,1) 正态分布的数
# 2.5*np.random.randn(2,4) +3  # 返回一个(3, 2.5*2.5)的正态分布
# =============================================================================
# 2.数据类型
# =============================================================================
# dtype的使用, 可以参考https://blog.csdn.net/kaever/article/details/68945075
# numpy 包含的数据类型有：int8、16、32、64; uint8、16、32、64; float16、32、64
# 以及complex64、128

student = np.dtype([("name", 'S10'), ("age", 'int8'), ('mark', 'f4')])
b1 = np.array([('abc', 18, 30), ('def', 19, 50)], dtype=student)
print(b1["name"])  # 将数据类型为name中筛选出来

# =============================================================================
# 3.Numpy 数组常用属性
# =============================================================================

c1 = np.ones((2, 4), dtype="int16")
# c1.ndim   # c1所在空间的维度，输出为2,即为2维的
# c1.shape  # c1的维数，输出为（2，4）,类型为tuple
# c1.dtype  # 数据类型
# c1.itemsize  # 以字节的形式输出每项的大小，这里为2，因为int16占了2个字节
# c1.size  # 输出为8，即为2*4=8
# c1.real  # 输出实部
# c1.imag  # 输出虚部
# c1.flags # 输出c1相应的内存信息
# c1.data  # 输出c1在缓冲区中的地址

# =============================================================================
# 4.从已有数组创建数组
# =============================================================================
#
# 主要有asarray frombuffer fromiter
# asarray
d1 = np.full((3, 4), 9)
e1 = np.asarray(d1, dtype='float')
# frombuffer  通过流来创建数组
str1 = b"hello python"
f1 = np.frombuffer(str1, dtype='S1')
f2 = np.frombuffer(b'\x01\x02\x03\x04', dtype=np.uint8, count=3)
# fromiter : 通过迭代来创建数组
list1 = np.linspace(1, 10, 10)
it1 = iter(list1)  # 创建迭代对象
f3 = np.fromiter(it1, dtype='float', count=-1)

# =============================================================================
# 5.从数组范围内创建数组
# =============================================================================

# 主要有 arange linspace logspace
g1 = np.arange(1, 10, 1)  # 从1到9，步长为1
g2 = np.linspace(0, 20, 11)  # 从0到20 共11个数
g3 = np.logspace(1, 2, 5, base=16)  # 16的1次方 到16的2次方 5分之1倍程,默认倍数为10

# =============================================================================
# 6.切片与各种索引
# =============================================================================

# 主要有利用slice切片，直接切片
# 利用slice
h1 = np.arange(10)
sl1 = slice(2, 7, 2)  # 从索引2到索引7停止（不包括索引7），间隔为2，
h2 = h1[sl1]

# 直接切片
h3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
h4 = h3[..., 1]  # h3的第2列，返回为4*1的矩阵(即为1列)
h5 = h3[1, ...]  # h3的第2行
h6 = h3[..., 1:]  # h3的第2列以及剩下所有元素
h7 = h3[1:]  # h3 的第2行以及剩下的所有元素

# 各种切片/索引
h8 = h3[[0, 1, 2], [1, 0, 1]]  # 通过组合输出 [0,1] [1,0] [2,1] 三个数组成的数列

rowTemp1 = np.array([[0, 0], [3, 3]])
colTemp1 = np.array([[0, 2], [0, 2]])
h9 = h3[rowTemp1, colTemp1]  # 返回h3四个角上的元素，2*2的矩阵

h10 = h3[h3 > 2]  # 返回h3中大于2的数并组成1列

h11 = np.array([np.nan, 1, 2, 3, np.nan])
h12 = h11[~np.isnan(h11)]  # 通过求补符号~返回h11 中所有非NaN的元素

h13 = np.array([1+1j, 2, 3+2j])
h14 = h13[np.iscomplex(h13)]  # 返回所有非复数的数组成一列

h15 = h3[np.ix_([1, 2, 0, 1], [2, 1, 2, 1])]  # 通过调用np.ix_ 选择指定的数，返回4*4的矩阵

# =============================================================================
# 7.矩阵值运算(可以扩展为numpy 里的 广播broadcast)
# =============================================================================
#
# python 中的矩阵值和matlab中不同，不需要加.符号，直接运算就是矩阵值运算，而且 当其
# 中一个矩阵是n*1 的， 其余的矩阵是n*m的, 那么该矩阵的所有列 都会被加上该n*1的矩阵
# 这就是所谓的broadcast
k1 = np.array([[1, 2, 3], [4, 5, 6]])
k2 = np.array([[4, 5, 6], [7, 8, 9]])
k3 = k1 + k2

k4 = np.ones((1, 3))
k5 = k1 + k4   # k5 = array([[2., 3., 4.], [5., 6., 7.]])

# =============================================================================
# 8. 关于迭代
# =============================================================================

# 和Python里自带类型--列表不一样，numpy里的数组/矩阵，并不能直接参与迭代，它需要
# np.nditer处理后才能参与迭代
# 8.1 常用迭代
m1 = np.array([[1, 2, 3], [4, 5, 6]])
# with  np.nditer([a1, None], order='C') as it1: # C模式以行优先，F模式以列优先
it1 = np.nditer([m1, None], order='C')  # None 做为输出
for seq1, seq2 in it1:
    seq2[...] = seq1**2  # 这里的[...]很关键
    print(seq1)
m2 = it1.operands[1]  # 输出整个m1**2

# 8.2 允许修改自身的迭代
m2 = np.array([1, 2, 3, 4])
for seq3 in np.nditer(m2, op_flags=['readwrite']):
    seq3[...] = seq3*3
print(m2)  # 输出m2*3

# 8.3 外部循环的迭代：给出的值是具有多个值的一维数组，而不是零维数组
m3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for seq4 in np.nditer(m3, flags=['external_loop'], order='F'):
    print(seq4, end='\n')  # 每次输出一行而不是一个数

# 8.4 如果迭代的数组是两个或以上，且维数满足广播条件，就会触发广播
m4 = np.array([[1, 2, 3], [4, 5, 6]])
m5 = np.array([1, 2, 3], dtype='float')
for m4, m5 in np.nditer([m4, m5]):
    print(m4, 'and', m5)

# =============================================================================
# 9.常用数组操作
# =============================================================================

n1 = np.arange(10).reshape(2, 5)  # reshape的用法

# 9.1 修改数组形状
for element in n1.flat:  # flat的用法，做为数组单个元素的迭代器
    print(element)
n2 = n1.flatten(order='F')  # flatten的用法， 将数组展平，"C"是按行，"F"是按列
n3 = np.ravel(n1, order='F')  # ravel的用法， 也是将数列展平

# 9.2 翻转数组
n4 = n1.T  # n1的转置
n5 = np.transpose(n1)  # n1的转置

n6 = np.ones((1, 2, 3, 4))
n7 = np.rollaxis(n6, 3)  # rollaxis,将轴3向后滚动，这里是(1,2,3,4)滚成了(3,1,2,4)
n8 = np.rollaxis(n6, 1, 3)  # 滚成了(1,3,2,4), 轴的顺讯是从0开始的

n9 = np.swapaxes(n6, 0, 3)  # 交换两根轴的位置， 变成了(4,2,3,1)

# 9.3 改变数组维度
# 还有函数 broadcast broadcastto
n10 = np.ones((2, 2))
n11 = np.expand_dims(n10, 0)  # 从2*2 变成了 1*2*2
n12 = np.squeeze(n11)  # 又从1*2*2 变成了2*2

# 9.4 连接数组
n13 = np.ones((3, 4))
n14 = np.zeros((2, 4))
n15 = np.concatenate((n13, n14), axis=0)  # 把n13和n14连接起来，形成5*4的矩阵

n16 = np.full((3, 4), 2)
n17 = np.stack((n13, n16), axis=0)  # 沿着0轴将n13和n16堆叠起来
# 此外还有vstack 和 hstack 即垂直堆叠和水平堆叠

# 9.5 分割数组
n18 = np.split(n13, [2, 3], axis=1)  # 利用split 将数组n13拆分成几个行组成的list
# 此外还有hsplit（水平拆分） 和 vsplit(垂直拆分)

# 9.6 元素的添加与删除

# 有append() insert() delete() resize() uique() 等函数

# =============================================================================
# 10. 位运算
# =============================================================================
#
# 有 np.bitwise_and ()  np.bitwise_not()   np.bitwise_or()
# 以及 np.bitwise_xor()   np.invert() 等函数

# =============================================================================
# 11. 字符串函数
# =============================================================================
#
# 有 np.char.add()   np.char.join()    np.char.split()  np.char.strip():去头去尾
# np.char.capitalize()
# np.char.lower() np.char.upper() np.char.title() np.char.replace()
# np.char.decode():code到字符  np.char.encode(): 字符到code 等函数

# =============================================================================
# 12. 常用数学函数
# =============================================================================
#
# 三角函数相关： np.sin()  np.cos()  np.tan()  np.arccos()  np.arcsin()  
#              np.arctan() np.arctan2 () np.deg2rad() np.degrees()

# 四舍五入相关：np.round()  np.ceil()  np.floor()

# 算术相关：np.add()  np.subtract()  np.multiply()  np.divide()  np.reciprocal():倒数
#          np.power():幂  np.mod():余数  直接 + - * / ** % 也可以

# 统计相关：
# np.amax()  np.amin()  np.mean()  np.ptp(): 最大值与最小值的差值
# np.percentile(): 百分比对应的（数组中百分之几的数都返回的数要小）
# np.median():中位数   np.average(): 加权平均值
# np.std(): 标准差 np.var(): 方差

# 排序和条件筛选函数
# np.sort() : 可以选择排序算法，以及有选择的排序
# np.argsort() : 返回排序的序号
# np.lexsort() : 对多个序列进行排序
# np.argmax()  np.argmin() : 返回最大/最小值得索引
# np.sort_complex() : 对复数进行排序，先实部，后虚部
# np.where() : 返回符合条件的值得索引
# np.extract(): 抽取符合条件的值

# =============================================================================
# 13. numpy中的副本与视图
# =============================================================================

# Python中直接赋值是给与变量以别名，和原数值享有相同的地址
# 视图是对原对象的一个引用，修改视图有可能对原对象产生影响，类似于浅拷贝
# 深拷贝才是真正的拷贝， 原对象和拷贝产生的对象之间不再具有联系

# Python的切片是深拷贝, 但是numpy中的切片只是视图！！！
# numpy中的ndarray.view() 相当于 python中 list.copy()
# numpy中的ndarray.copy() 相当于 python中的 copy.deepcopy()！！！

# 所以，列表要习惯用deepcopy， numpy要习惯用copy!!

# =============================================================================
# 14. numpy中的矩阵库
# =============================================================================

# numpy 中有专用的矩阵库, matlib, 需要引用 import numpy.matlib

# numpy 中的矩阵和数组可以相互转化 通过np.asarray()  np.asmatrix()

# matlib自带的函数有
# np.matlib.zeros()  np.matlib.ones()  np.matlib.eye()  np.matlib.full()
# np.matlib.identity(): 单位矩阵
# np.matlib.rand()

# 矩阵的*是 矩阵乘

# 矩阵运算的函数有(同样用于ndarray)：
# np.dot():内积  np.vdot()  np.inner()  np.matmul():矩阵相乘

# numpy 还自带线性代数的库 linalg
# np.linalg.det(): 行列式
# np.linalg.solve(): 解方程
# np.linalg.inv():求逆
np.linalg.cholesky
# =============================================================================
# 15.Numpy中的IO
# =============================================================================

# 主要是save 和 load, 与matlab的使用类似

# np.save()  np.savez()：保存多个文件， 保存的格式都为npy
# np.load()：载入文件
# np.savetxt: 保存为txt文件
# np.loadtxt: 读取txt文件







