# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:34:59 2019

@author:  HuangLingZhi
Email : hlzcow1986@126.com
# : 欢迎转载:转载请注明出处
# Just used for pandas study
"""

#import wx
#app = wx.App()
#frm = wx.Frame(None, title="第一个wxPython")
#frm.Show()
#app.MainLoop()
#import numpy as np
#img1 = np.zeros((20, 20)) + 3
#img1[4:-4, 4:-4] = 6
#img1[7:-7, 7:-7] = 9
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # 3d坐标轴
# =============================================================================
# 1. 常用线图
# =============================================================================

# 1.1 柱状图
#plt.figure(1)
#plt.bar(np.linspace(1,10,10),np.random.randint(1,10,10))
#plt.ylabel('SOME')
#plt.xlabel('hehe')
#plt.title('titel1')
#
# 1.2 线图
#plt.figure(2)
#plt.axis([0,16,-3,3])
#plt.grid(True)
#plt.title('title-')
#plt.plot(np.linspace(0,10,10), np.random.randn(10))
#plt.xlabel('x-')
#plt.ylable('y-')

# 1.3 子图
#plt.figure(3)
#plt.subplot(2,1,1)  # 两行一列，第一个图
#plt.plot(np.arange(0,11,2), np.arange(0,11,2))
#plt.subplot(2,1,2)  # 两行一列，第二个图
#plt.plot(np.arange(0,11,2),np.ones(6))

# 1.4 散点图
#plt.figure(4)
#x1 = np.random.normal(0,1,1024)
#y1 = np.random.normal(0,1,1024)
#z1 = np.arctan2(x1,y1)
#plt.scatter(x1,y1, c=z1, alpha=0.5) # 散点图

# =============================================================================
# 2. 常用三维/投影图
# =============================================================================

# 2.1 曲面图和等高线图
fig5 = plt. figure(5)
x1 = np.linspace(-3, 3, 100)
y1 = np.linspace(-3, 3, 100)
X1, Y1 = np.meshgrid(x1, y1)

def f(x,y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

Z1 = f(X1, Y1)
#plt.contour(X1, Y1, f(X1, Y1))  #等高色图，contourf是等高线图
ax = Axes3D(fig5)
ax.plot_surface(X1, Y1, Z1, cmap=plt.get_cmap('rainbow'))  # 3维色图
ax.contourf(X1, Y1, Z1,zdir='z',offset=-2, cmap=plt.get_cmap('rainbow'))
ax.set_zlim(-2,2)

# 


