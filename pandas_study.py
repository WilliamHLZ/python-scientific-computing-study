# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:02:56 2018

@author: William

# Just used for pandas study
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# =============================================================================
# 1. 对象的创建
# =============================================================================

# 1.1 serials的创建
a1 = pd.Series([1, 2, 3, np.nan, 4])

# 1.2 DateFrame的创建
# 创建方式1
dates = pd.date_range('20071130',  periods=12, freq='12M')
a2 = pd.DataFrame(np.random.randn(12, 6), index=dates, columns=list('ABCDEF'))

# 创建方式2
a3 = pd.DataFrame({'A': 1,
                   'B': pd.Timestamp('20181011'),
                   'C': pd.Series(1, index=list(range(4)),
                                  dtype='float32'),
                   'D': np.array([3]*4, dtype='int32'),
                   'E': pd.Categorical(['test', 'train',
                                        'test', 'train']),
                   'F': 'fool',
                   'G': [1, 2, 3, 4]}, index=list('abcd'))
#
# =============================================================================
# 2.查看数据
# =============================================================================

a2.head()  # 查看a2顶部的数据
a2.tail(3)  # 差看a2尾部3行的数据
# 查看索引、列标、numpy数据
a2.index
a2.columns
a2.values
#
a2.describe()  # 快速统计

a2.T  # a2的转置
a2.sort_index(axis=0)  # 根据指定轴进行排列
a2.sort_values(by='B')  # 根据指定列进行排列

# =============================================================================
# 3.选择
# =============================================================================

a2['A']  # 等同于a2.A  返回一个Series，修改会影响原对象
a2[0:3]  # 对行进行切片
a2['20071130': '20101130']  # 对行进行切片

a2.loc['20071130']
a2.loc['20071130', 'A': 'C']
a2.iloc[1]  # 对第2行做切片，和a2[1]一致
a2.iloc[0:3, 0:4:2]  # 不包括第3行，第4列
# ------------------ # Dateframe 不能直接[0:3, 0:4:2]这样切片，
# ------------------ # 但List可以，所以a2.vaules[0:3, 0:4:2] 也是可以的
a2.loc[dates[0], 'A']  # dates是a2的index, 这里返回一个标量,且和原数据脱钩
a2.at[dates[0], 'A']  # 和上式等价

a2[a2['A'] > 0]  # 通过这'A'列对数据进行筛选
a2[a2 > 0]    # 所有非0元素显示为NaN

a3['E'] = ['test', 'train']*2  # 变成了4列的向量
a2[a2['E'].isin(['test'])]

# 设置
c1 = pd.Series(list(range(1, 13, 1)), index=a2.index)
a2.loc[:, 'G'] = c1  # 设置一个新的列
a2[dates[0], 'H'] = 1234  # 设置一个新的值

a2.loc[:, ['D']] = np.array([5] * len(a2.index))
a2[a2 > 0] = -a2  # 通过where语句来设置

# =============================================================================
# 4.对缺失数据的处理
# =============================================================================

# 通过已存在的dateframe新建一个（返回一个和原始无关的）
d0 = a2.reindex(index=a2.index[0:3], columns=(list(a2.columns) + ['A2']))
d1 = d0.copy()

d1.loc[0:2, 'A2'] = [2]*2

d2 = d1.fillna(value=123)  # 将含有nan的值变为123
d3 = d1.dropna(how='any')  # 删除含nan的行

pd.isnull(d1)  # 返回一个和原dataframe相同维数的true false的文件
# 即布尔填充

# =============================================================================
# 5. 相关操作
# =============================================================================

a2.mean(1)  # 对轴1进行平均

s2 = pd.Series(np.ones(len(a2.index)), index=a2.index)

e1 = a2.sub(s2, axis='index')  # 将减法自动广播到e1的所有列上
a3 = -a2
e2 = a3.apply(np.log10)  # 将np.log10() 运用到e2的所有数据上，也可以运用
# -----------------------# 自己定义的函数

s3 = pd.Series(np.random.randint(0, 10, size=10))
e3 = pd.value_counts(s3)  # 统计出现次数

e4 = pd.Series(['Ac', 'C', 'a', 'b'])
e4.str.join('.')  # 用.连 # .str中还有大量的字符串方法

# =============================================================================
# 6. 合并
# =============================================================================

f1 = pd.DataFrame(np.random.randint(1, 11, size=(10, 3)))
f2 = [f1[0:3], f1[3:6], f1[6:10]]  # 这种切片依然受到f1的影响
f3 = pd.concat(f2)  # 合并后返回一个独立的值

f4 = pd.DataFrame({'1st': [1, 3], '2nd': [4, 5]})
f5 = pd.DataFrame({'1st': [1, 3], '2nd': [5, 5]})
f6 = f4.merge(f5, how='outer')  # 合并，可以合并f4和f5的同类项

f4.index = ['a', 'b']
f5.index = ['c', 'd']
f7 = f4.append(f5, ignore_index='true')  # 将f5直接合并到f4下面，并忽略index
f7 = f7.append(f7.iloc[1],  ignore_index='true')  # 将f7的第一行添加到到f7底下

# =============================================================================
# 7. 分组
# =============================================================================

g1 = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'bar'],
                   'B': ['good', 'great', 'excellent', 'perfect',
                         'good', 'great', 'excellent', 'perfect'],
                   'C': np.random.randn(8),
                   'D': np.random.randint(8, size=8)})

g2 = g1.groupby('A').sum()  # 对G1的分类然后求和
g3 = g1.groupby(['A', 'B']).sum()  # 对A,B两类的分类求和
# ---------------------------------#  g3的Index是('foo', 'bar') 等等

# =============================================================================
# 8. 改变形状
# =============================================================================
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

# 注8.1：在调用函数传递参数时，可以在实参序列前加一个星号*进行序列解包，
# ------或在实参字典前加两个星号**进行解包。
# ------或在实参字典前加两个星号**进行解包。
# 注8.2：zip函数 zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包
# ------成一个个元组，然后返回由这些元组组成的列表。元素个数与最短的列表一致
# nums = ['flower', 'flow', 'flight']
# 如：for i in zip(*nums):
#      print(i)
# 系统将输出 ('f', 'f', 'f')
# ('l', 'l', 'l')
# ('o', 'o', 'i')
# ('w', 'w', 'g')

index1 = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
g4 = pd.DataFrame(np.random.randn(8, 2), index=index1, columns=['A', 'B'])
# 显示为具有两个元素的index
g4.xs('one', level=1)  # 显示分索引(level=1即'second')下的one
g4.loc[(slice(None), 'one'), :]  # 同上式
g4.loc[(slice('bar', 'baz'), 'one'), 'A']  # 注意切片器slice的用法
g4.loc(axis=0)[:, 'one'] = 0  # 设置某些值为零
#
g4_stacked = g4.stack()
#
index2 = pd.MultiIndex(levels=[['I', 'II'], ['i', 'ii']],
                       labels=[[0, 0, 1, 1], [0, 1, 0, 1]],
                       names=['One', 'Two'])
g5 = pd.DataFrame(np.random.randn(4, 4), index=index2)
g6 = g5.mean(level=0)

#  关于index的一些操作
g6 = g6.reindex(g5.index, level=0)  # 重新定义index, 并用原level0的数据去补全
g5 = g5.reset_index()  # 所有的index被重置成新列
g5 = g5.set_index('One')  # 重新将g5的index设置为One
#
# 8.2节 数据透视图
g7 = pd.DataFrame({'I': ['i1', 'i2', 'i3', 'i4']*3,
                   'II': ['ii1', 'ii2', 'ii3']*4,
                   'III': ['iii1', 'iii2']*6,
                   'value1': np.random.randint(100, size=12),
                   'value2': np.random.randn(12)})
#
g7.pivot_table(values='value1', index=['I', 'II'], columns='III',
               aggfunc=np.max)  # 数据透视表, 聚合函数选用aggfunc=npmax
g7.pivot(values='value1', columns='III', index='value2')

# =============================================================================
# 9.时间序列
# =============================================================================

# 9.1 Pandans 时间相关类以及创建方法有：
# Timestamp(时刻数据)： to_datetime Timestamp
# DatetimeIndex(Timestamp的索引):  to_datetime daterange DateTimeIndex
# Period(时期数据)： Period
# PeriodIndex(PeriodIndex)： periodrange PeriodIndex
# 还可以参考比对一下一下 Python自带的datetime类

# to_datetime的用法实例：
h1 = pd.to_datetime(['20101112080033', '2010-12-1', '2010-12-1-08-00'])
# Timestamp 返回一个时间戳
h2 = pd.Timestamp('2017-1-1-12-00-00')
h3 = pd.to_datetime('10-11-2099 18:00:05', format='%d-%m-%Y %H:%M:%S')

# DatetimeIndex的实例
startT = datetime.datetime(2012, 1, 1)
endT = datetime.datetime(2017, 12, 1)
index4 = pd.date_range(startT, endT, freq='45D')  # 45天的间隔的DateTimeindex
h4 = pd.Series(np.random.randn(len(index4)), index=index4)
h4['2012']  # 可以通过部分时间戳对数据进行引用

# 9.2 典型例子
index5 = pd.date_range('2011-11-1', periods=100, freq='Y')
h5 = pd.Series(np.random.randn(len(index5)), index=index5)
h5.resample('M').sum()  # 以分为间隔重新统计h5
h6 = h5.resample('D').asfreq()[0:5]  # 重新统计h5,并将前5行存入h6

# 9.3 时区相关
h7 = h6.tz_localize('UTC')  # 将时区固定为utc
# UTC：格林尼治时间，全世界唯一的统一时间；
# import pytz # 第三方时间库
# 通过pytz.common_timezones可以获得所有时区的名称
# import dateutil # dateutil目前仅支持固定偏移和tzfile区域。
h8 = h7.tz_convert('Asia/Shanghai')  # 时区转换
# h8 = h7.tz_convert(dateutil.tz.tzoffset('UTC', 8*60))  # 北京时间时区转换

# 9.4 时间的一些运算
h9 = pd.period_range('20121101', periods=100, freq='D')
h10 = pd.Series(np.random.randn(len(h9)), index=h9)
h10 = pd.DataFrame(h10)
h11 = h10.copy()
h11.index = (h9.asfreq('Y', 's')+1)  # .asfreq('H','s') +9
# 改变排列显示，并在基础上+1, 's'代表start, 'e'代表end

# =============================================================================
# 10.Categorical
# =============================================================================

i1 = ['a', 'a', 'b', 'b', 'c']
i2 = pd.Categorical(i1)  # 建议一个categorical类型

i2.categories  # 返回i2中的种类 Index(['a', 'b', 'c'], dtype='object')
i2.codes  # 返回种类对应的序号  array([0, 0, 1, 1, 2], dtype=int8)

i3 = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
# 给i3增加一列category类型的数据
i3["grade"] = i3["raw_grade"].astype("category")
# 重新定义i3的category, 并可以通过i3.grade.cat.categroies观察其特性的变化
i3["grade"].cat.categories = ["very good", "good", "very bad"]
# 再次设置i3.grade的属性，使其增加一些选项
i3["grade"] = i3["grade"].cat.set_categories(["perfect",
                                              "very good", "good", "very bad"])
# 根据grade中catagory对字典进行排序
i3.sort_values(by='grade')
# 根据grade进行排序
i3.groupby('grade').size()

# =============================================================================
# 11. matplotlib
# =============================================================================

j1 = pd.Series(np.random.randn(1000),
               index=pd.date_range('2010-1-1',  periods=1000, freq='2D'))
#
j1 = j1.cumsum()  # 累计求和　
#
plt.figure()
j1.plot()
plt.legend(loc='best')
#
j2 = pd.DataFrame(np.random.randn(1000, 4),
                  columns=['A', 'B', 'C', 'D'],
                  index=j1.index)
j2.cumsum().plot()
plt.legend(loc='best')

# =============================================================================
# 12. 数据文件的读写
# =============================================================================

k1 = pd.DataFrame(np.random.rand(10, 10), columns=list('abcdefghij'),
                  index=list(range(0, 10, 1)))
# 12.1 读写csv文件
k1.to_csv('filename1.csv')
k2 = pd.read_csv('filename1.csv')
# 12.2 EXCEL读写
k1.to_excel('filename1.xlsx', sheet_name='sn1')
k3 = pd.read_excel('filename1.xlsx', sheet_name='sn1', na_values=['NA'])
# 12.3 hdf5文件的读写
k1.to_hdf('filename1.h5', 'k1')
k4 = pd.read_hdf('filename1.h5', 'k1')





















