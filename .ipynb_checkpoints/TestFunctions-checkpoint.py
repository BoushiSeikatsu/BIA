#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np

# In[6]:


#x is point in graph, that has d dimensions
def Ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = len(x)
    squareSum = 0
    cosSum = 0
    for i in x:
        squareSum += pow(i,2)
    for i in x:
        #print(i)
        cosSum += np.cos(c*i)
    return -a * np.exp(-b * np.sqrt(1/d*squareSum)) - np.exp(1/d * cosSum) + a + np.exp(1)

def Rastrigin(x):
    d = len(x)
    totalSum = 0
    for i in x:
        totalSum += pow(i,2) - 10 * np.cos(2*math.pi*i)
    return 10*d*totalSum

def Sphere(x):
    totalSum = 0
    for i in x:
        totalSum += pow(i,2)
    return totalSum

def Rosenbrock(x):
    totalSum = 0
    for i in range(0, len(x)-1):
        totalSum += 100 * pow(x[i+1] - pow(x[i],2),2) + pow(x[i] - 1,2)
    return totalSum

def Griewank(x):
    totalSum = 0
    totalMul = 0
    k = 1
    for i in x:
        totalSum += pow(i,2)/4000
        totalMul *= np.cos(i/np.sqrt(k))
        k += 1
    return totalSum - totalMul + 1

def Schwefel(x):
    d = len(x)
    totalSum = 0
    for i in x:
        totalSum += i * np.sin(np.sqrt(abs(i)))
    return 418.9829*d - totalSum

def Levy(x):
    d = len(x)
    totalSum = 0
    for i in range(0,d-1):
        w = 1 + (x[i] - 1)/4
        totalSum += pow(w - 1,2) * (1 + 10 * pow(np.sin(np.pi*w + 1),2))
    w1 = 1 + (x[0] - 1)/4
    wd = 1 + (x[-1] - 1)/4
    return pow(np.sin(np.pi*w1),2) + totalSum + pow(wd - 1,2) * (1 + pow(np.sin(2 * np.pi * wd),2))

def Michalewicz(x):
    d = len(x)
    m = 10
    totalSum = 0
    k = 1
    for i in x:
        totalSum += np.sin(i)*pow(np.sin((k * pow(i,2))/np.pi),2*m)
        k += 1
    return -1 * totalSum

def Zakharov(x):
    d = len(x)
    firstSum = 0
    secondSum = 0
    k = 1
    for i in x:
        firstSum += pow(i,2)
        secondSum += 0.5*k*i
        k += 1
    return firstSum + pow(secondSum,2) + pow(secondSum,4)

#Returns functions in form of list of tuples (minX, maxX, minY, maxY, func)
def getAllFunctions():
    funcs = [(6, -6, -10, 10, Sphere), (40, -40, -40, 40, Ackley), (5, -5, -5, 5, Rastrigin), (6, -6, -10, 10, Rosenbrock), (5, -5, -5, 5, Griewank), (500, -500, -500, 500, Schwefel), (10, -10, -10, 10, Levy), (4, 0, 0, 4, Michalewicz), (10, -10, -10, 10, Zakharov)]
    return funcs