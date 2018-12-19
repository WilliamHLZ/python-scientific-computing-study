"""

@author: William
email : hlzcow1986@126.com
# : 欢迎转载:转载请注明出处
# Just used for python study
"""

import sys
import os
import import_test # 尝试导入自建的模块
import pickle
import io
import time
import this  # 输出：对python编程的说明和建议
from math import sqrt
# 从math模块中导入sqr,使得本模块中可以直接使用sqrt，若非如此，则需使用math.sqrt();
# 但本方法不建议使用还可以用import **as ** 或者 from ** import ** as
# =============================================================================
# 0. 程序说明
# =============================================================================
# First python programm
# 本程序用于python 入门
# ipython 里的常用命令 1.? 返回ipython简介  2.quickref 返回ipython摘要 3.obj?以及obj?? 和help()类似
#                     4.%magic 魔方魔法 5.!命令 感叹号加命令 即使用系统命令，如：!ping !cmd
#                     6.Table 可以自动补全
# spyder 里的常用快捷键：1.ctrl+1 单行注释/取消注释 2.ctrl+4/5 添加/取消块注释 3.shift+tab: 调出函数的Arguments
#                       4. ctrl+F/R 查找/替换文本  5.ctrl+i 调出帮助文档
#                       6. ctrl+鼠标左键 进入该函数 7.F5 运行  8.F9 运行选中的区域
#                       9. ctrl+shift+v 调出变量窗口
# =============================================================================
#1. 实验print函数的用法
# =============================================================================
print("hello world!")
print(""" "hello!" "hi!"  """)
a1 = 'william'
b1 = 2
c1 = 2.12345
print('{0} is not \n {1}'.format(a1,b1))
print('{0:<+04.3f} is a formatnume'.format(c1) ) #左对齐,加正号,宽度4，小数点后3, 浮点型
print('{0:#b},{0:#o},{0:#x}'.format(16)) # 数字16 的 2进制，8进制, 16进制表示
print('hehe' +' '+ '2')
#a1= str(input("hehe"))
print('hehe3'+ a1)
# =============================================================================
#2. 实验定义函数以及if和for以及while结构体
# =============================================================================
def fun1_test(a1 = 1,b1 = 2, c1 = 3):
    '''
    函数说明段
    :param a1:
    :param b1:
    :param c1:
    :return:
    '''
    if a1>2:
        print(a1)
    elif a1<-1:
        print(b1)
    else:
        print(c1)
#
fun1_test(3)
def fun2_test(a1 = 2, *param1, **dictionary):
    '''
    不同的函数输入试验
    :param a1: 数字1
    :param param1:数列1
    :param dictionary:字典1
    :return: none
    作为测试的函数2
    '''
    print("a1=",a1)
    for seq1 in param1:
        print(str(seq1)+'=',seq1)
    for seq2,seq3 in dictionary.items():
        print(seq2,seq3)
#
fun2_test(3,1,2,3,4,5,william=1,jack=2,mary=3)
print(fun2_test.__doc__) # 返回说明文档 ，或者用help(fun2_test2)
# 试验for..in 以及 while 结构体
def fun3_test(a1):
    for seq1 in sys.argv:
        print(seq1)
        print(sys.path)
    while a1>=1 and a1<5:
        print('a1确实大于等于1,a1={0}'.format(a1))
        a1+=1
fun3_test(1)
print('path：',os.path)
print('path：',os.getcwd())
#
import_test.fun_import_test1() # 实验 导入模块import_test中函数的使用
#
print(dir()) # 输出本模块的所有变量、函数名等
del a1  # 删除a1
print(dir())

# 2.2 返回多个值，只要用一个元组就可以了
def fun1(a, b):
    d = a+b
    e = a-b
    return(d, e)

d1,e1 = fun1(2,1)

# =============================================================================----------
# 3.现在开始实验列表(list)
# =============================================================================----------
# 常用的数据结构有 列表:用[]表示  元组：用()表示  字典：用{}表示
# 3.1 列表的基本用法
test_list1 = [2,4,3] # 原始列表
test_list1.sort()

del(test_list1[1]) # 删除元素方法1

test_list1.append(5) # 加入元素

test_list1.remove(test_list1[0]) # 删除元素方法2

test_list1.append('hehe') # 加入不同类型的元素

test_list1.insert(0,'HaHa') # 插入元素


# 3.1 列表推导式
#　即用[表达式 for 变量 in 列表 （可选if条件）]
lst1 = [3,4,5,8]
lst2 = [(a1,b1) for a1 in lst1 if a1<4 for b1 in lst1 if b1>=4]



# =============================================================================----------
# 4.现在开始实验元组(tuple)
# =============================================================================----------
# 注：元组内部元素不可更改
test_tuple1 = ('first','second','third') # 元组1
print('length is',len(test_tuple1))
#
test_tuple2 = ('fourth','fifth')  # 元组2
print('length is ',len(test_tuple2) )

test_tuple3 = (test_tuple1,test_tuple2)  # 元组3
print(test_tuple3,end= 'length is '+ str( len(test_tuple3) ))

print('\n'+ str(len(test_tuple3[1])) )
test_tuple4 = ('hehe',) # 只有一个元素的元组表示
print(test_tuple4) #
test_tuple5 = () # 没有元素的元组表示
print(test_tuple5) #
#
# =============================================================================----------
# 5.现在开始实验字典(dictionary)
# =============================================================================----------
# 字典由键值(keys)和值(values)联系到一起，字典不会以任何形式排序
test_dic1 = {'first': 1, 'second': 'hehe2', 'third':'haha '}


def printdic():
    '''
    打印字典1的元素
    '''
    print('字典1的元素有')
    for name,vaule in test_dic1.items():
        print(name+':'+ str(test_dic1[name]))
printdic()

del(test_dic1['second'] )
printdic()
test_dic1['fifth'] = 8
printdic()
#
if 'fifth' in test_dic1:  # 查找元素是否已在python里
    print('元素已经加入字典里啦')
#

# 创建空字典(没有具体数值，只有名目)
test_dic2 = {}.fromkeys(['item1', 'item2'])

# =============================================================================
# 6.现在开始实验序列
# =============================================================================
# 列表(list) 元组(tuple)以及字符串(string)都是序列
# 序列可以用切片操作，和matlab类似，但实现细节上有一些不同
test_list3 = ['1st', '2nd', '3rd', '4th', '5th']  #这是一个列表
print(test_list3[0:])  # 全部元素
print(test_list3[ :])  # 全部元素
print(test_list3[0:2])  # 只包括第[0]个，而不包括第[2]个
print(test_list3[0:-1])   # -1 代表最后一个元素，-2代表倒数第二个元素
print(test_list3[0::2])  # 以2为步长的序列，这一点和mabtlab不同，呵呵
# =============================================================================
# 7.集合(collection)
# =============================================================================
# 集合是无序的 集合可以用in 来判断 集合中是否包含某元素
# 集合可以用 & 表示交集 ，可以用add remove 添加和删除元素
test_col1 = set([3, 5, 'apple', 'pear'])
test_col2 = set([5, 'pear', 'banana'])
test_col1.add('banana')
test_col2.remove(5)
print(test_col1.intersection(test_col2)) # 交集
print('banana' in test_col1)
print(test_col1.issubset(test_col2)) # 判断是否子集
# =============================================================================
# 8.字符串(string)
# =============================================================================
# 字符串也是一种对象，有许多实用的命令
str1 = 'abcdef'
str2 = 'defghi'
print(str1.startswith('cb'))  # 判断str1是否以cb开头
print( 'def' in str1 )  # 判断def是否在str1内
print(str1.find('cd'))  # 返回查找的位置，如果不在则返回-1
# =============================================================================
# 9.实际问题的编程，请参见zip_test.py
# =============================================================================
#
# =============================================================================
# 10.关于类(class)
# =============================================================================
# 10.1 关于方法和变量
#  类中的方法有三种，即普通方法，类方法(用@classmethod修饰)，静态方法(用@staticmethod修饰)


class ClassTest:
    name = '1'
    nums = 0

    def __init__(self, name):  # __init__是构造函数不管有没有参数，定义函数都要用self做参数，
        self.name = name
        # ClassTest.nums += 1  # 类变量
        self.__class__.nums +=1  # 类变量 和 self.ClassTest.nums += 1 等价

    def fun1(self):  # 普通方法
        print('这个类的name是',self.name)
        print('对象变量nums', self.nums)

    @classmethod   # 类方法
    def fun2(cls):  # 类方法第一个参数可以用cls作为参数，即这个类(非实例)

        print('class_num的实例已达到',cls.nums)

    @staticmethod  # 静态方法
    def fun3():
        print('这是一个类方法，不需要输入self,也不需要输入cls')


# 类的引用
classA = ClassTest('William')
classB = ClassTest('Moranda')

# 类方法的引用
ClassTest.fun2()
classA.fun2()

# 普通方法的使用
classA.fun1()
ClassTest.fun1(classA)


# 10.2 关于继承
class ClassTest2(ClassTest):  # 继承ClassTest

    def __init__(self, name, second_name):
        ClassTest.__init__(self , name)  # 手动启动继承的构造函数
        self.secondname = second_name

    def fun1(self):
        ClassTest.fun1(self)
        print('这个类的第二名字是',self.secondname)


classC = ClassTest2( 'name1' , 'name2')
classC.fun1()

# =============================================================================
# 11.输入输出
# =============================================================================
# 11.1 input 函数


def fun_reverse(text1):
    text1 = text1[::-1]
    return text1

def ispar(text1):
    return text1 == fun_reverse(text1)


#something = input('请输入内容')
something = 'adda'

if ispar(something):
    print('{}是对称序列'.format(something))
else:
    print('{}是不对称序列'.format(something))

# 11.2  read 和 write 函数

poem = '''\n come on! Tomorrow will be better! '''

# 打开文件
f1 = open('c:\\test1.txt', 'a')
# 将poem写入文件
f1.write(poem)
# 关闭文件流
f1.close()

f1 = open('c:\\test1.txt')

while True:
    # 按行读文件
    line = f1.readline()

    if len(line) == 0:
        # 如果是空行，结束循环
        break
    else:
        print(line)

f1.close()

# 11.3  关于unicode 和 utf-8 , utf-8 广泛用于互联网传播中， 是unicode的一种编码方式
f1 = io.open('c:\\test1.txt','w',encoding='utf-8')
f1.write('今天天气不错')
f1.close()

f1 = io.open('c:\\test1.txt','r',encoding='utf-8')
print(f1.read())
f1.close()

f1 = io.open('c:\\test1.txt','r')
print(f1.read())  # 对比一下没有用相应编码读出的，将会是乱码
f1.close()

# =============================================================================
# 12.pickle模块 用于将python数据在文件中读写
# =============================================================================
# 和 matlab 的 load以及save 类似


filename1 = 'd:\\test2.dat'


f2 = open(filename1,'wb')

test_data1 = [1, 2, 3, 4]
test_data2 = ['a1', 'b1']
pickle.dump((test_data1,test_data2),f2)


f2.close()
del test_data1

f2 = open(filename1, 'rb')
data_copy1, data_copy2 = pickle.load(f2)

print(data_copy1)
print(data_copy2)

f2.close()

# =============================================================================
# 13.try except 模块，错误信息的处理
# =============================================================================
# 13.1 try except


class ShortInputException(Exception):  # 继承Exception类作为定义新的抛出异常的准备
    def __init__(self, length, atleast):
        Exception.__init__(self)
        self.length = length
        self.atleast = atleast


try:
    #test1= input('请输入一些内容')
    test1 ='123'
    if len(test1)<3:
        raise ShortInputException(len(test1), 3)

    print(test1*3)
except EOFError:
    print('You did a EOF operation! ')
except KeyboardInterrupt:
    print('You print Ctr+C')
except ShortInputException as ex:  # 注意as的用法
    print('The Input is {} long , The expected is at least {} '.format(ex.length, ex.atleast))
else:  # 不抛出异常情况下的执行
    print('{} is input'.format(test1))
finally:   # 不管是否抛出异常情况下也要执行
    print('whether except existed , program should continue')

# 13.2 with 语句
# 例子1：

class A(object):
    def __enter__(self):
        print('__enter__() called')
        return self

    def print_hello(self):
        print("hello world!")

    def __exit__(self, e_t, e_v, t_b):
        print('__exit__() called')


# 首先会执行__enter__方法
with A() as a:  # a为A.__enter__的返回对象
    a.print_hello()
    print('got instance')
# 结束会执行__exit__方法


# 例子2：
with open("test1.txt") as f:  # f等于open("test1.txt")的返回对象
    for line in f:
        print(line, end='')

# =============================================================================
# 14. 装饰器的用法，很有意思
# =============================================================================
#　使用时先定义装饰器函数，再使用装饰器函数
# 14.1 一般用法

def decoraFun(f_name):
    def inner(*args, **kwargs):
        print('\n这是首部装饰器')
        f_name(*args, **kwargs)
        print('这是尾部装饰器')
    return inner


@decoraFun
def f_ori(name1):
    print('这是原始函数'+name1)

#f_ori = decoraFun(f_ori)   #　这是老的调用方式，现在一般用@decoraFun表示装饰器

f_ori('haha')  # 调用函数时带了装饰器


# 14.2 类装饰器
class DecoratorC:
    def __init__(self,f1):
        self.f_name = f1

    def __call__(self, *args, **kwargs):  # __call__ 是类的特殊函数，使得一个类实例可以变为可调用对象
        print('\n这是类装饰器的首部')
        self.f_name()
        print('这是类装饰器的尾部')


@DecoratorC
def f_ori():
    print('这是原始函数')


f_ori()  # 调用函数时带了装饰器

# 14.3 装饰器链
#  即用多个装饰器来修饰原函数
def decoraFun2(f_name):
    def inner(*args, **kwargs):
        print('这是第二种首部装饰器')
        f_name()
        print('这是第二种尾部装饰器')
    return inner

@decoraFun
@decoraFun2
def f_ori():
    print('这是原始函数')

f_ori()
print(f_ori.__name__)

# 14.4 装饰器库 functool
# 使用wraps的目的是为了使得原来的函数加入了装饰器后，其名称和doc等属性不改变，
# 同时也不使用wraps可能出现的ERROR:  view_func...endpoint...map...

from functools import wraps, update_wrapper


def Deco(level):
  def decorator3(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('这是第{}级的装饰器库实现的装饰'.format(level))
        return func(*args, **kwargs)
    return wrapper
  return decorator3

@Deco(3)
def f_ori(name1):
    print('这是原始函数'+name1)


f_ori('haha')
print(f_ori.__name__)

# =============================================================================
# 15.1 python 内的常用内嵌函数(自带函数)
# =============================================================================

# 参考：http://www.runoob.com/python/python-built-in-functions.html
# 部分函数在python3.0 中已从全局函数中被移除，如 reduce() 被移至 fucntools 模块中

# abs()	divmod()	input()	open()	staticmethod()
# all()	enumerate()	int()	ord()	str()
# any()	eval()	isinstance()	pow()	sum()
# basestring()	execfile()	issubclass()	print()	super()
# bin()	file()	iter()	property()	tuple()
# bool()	filter()	len()	range()	type()
# bytearray()	float()	list()	raw_input()	unichr()
# callable()	format()	locals()	reduce()	unicode()
# chr()	frozenset()	long()	reload()	vars()
# classmethod()	getattr()	map()	repr()	xrange()
# cmp()	globals()	max()	reverse()	zip()
# compile()	hasattr()	memoryview()	round()	__import__()
# complex()	hash()	min()	set()
# delattr()	help()	next()	setattr()
# dict()	hex()	object()	slice()
# dir()	id()	oct()	sorted()	exec 内置表达式