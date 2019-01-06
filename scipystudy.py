# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 19:36:23 2018

@author: HuangLingZhi
Email : hlzcow1986@126.com
# : 欢迎转载:转载请注明出处
# Just used for scipy study
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
#  scipy study
# =============================================================================
# 1. optimize模块
# =============================================================================
#1.1 曲线拟合
#x1 = np.linspace(0,10, 100)
#
#
#def fun1(x1, a, b):
#   y1 = a*x1 + b
#   return y1
#
#y1 = fun1(x1, 1, 0) + np.random.randn(x1.shape[0])
#y2,erro1 = curve_fit(fun1, x1, y1)  # 函数的拟合
#
#plt.figure(5)
#plt.plot(x1, y1, x1, fun1(x1, y2[0], y2[1]))  # 拟合后的对比

#1.2 零点求解

#line_fun = lambda x: x+3
#y3 = fsolve(line_fun, 4)   # 求解零点，初始值选择 4
#
#def fun1(x):
#    return 2*np.sin(x*10)+0.5
#
#def fun2(x):
#    return x*0.1
#
#fun3 = lambda x: fun1(x)-fun2(x)
#
#x1 = np.linspace(-10, 10, 50)
#x2 = np.linspace(-10, 10, 1000)
#
#y3 = fsolve(fun3, x1)
#
#plt.plot(x2, fun1(x2), x2, fun2(x2))
#
#plt.scatter(y3, fun2(y3))

# =============================================================================
# 2 interpolate 插值模块
# =============================================================================

# 2.1 一维插值
x1 = np.linspace(0, 10*np.pi, 10)
y1 = np.sin(x1)

li1 = interp1d(x1, y1, kind='linear')
cu1 = interp1d(x1, y1, kind='cubic')

x2 = np.linspace(np.min(x1), np.max(x1), 100)
x3 = np.linspace(np.min(x1), np.max(x1), 100)
y2 = li1(x2)
y3 = cu1(x3)

plt.plot(x1, y1,'b')
plt.scatter(x2, y2, marker='o')
plt.scatter(x3, y3, marker= '^')
#plt.plot(x2, y2, marker='o') 
#plt.plot(x3, y3, marker= '^')
plt.contour()










