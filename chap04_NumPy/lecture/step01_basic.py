# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:04:31 2020

@author: user

"""
import numpy as np

lst=[1,2,3]

for i in lst:
    print(i**2)

arr = np.array(lst)
arr
arr**2 

#list -> numpy
arr = np.array([1,'two',3])
arr #array(['1', 'two', '3'], dtype='<U11')
arr.shape # (3,)

#동일타입
arr = np.array([[1,'two',3]])
arr #array([['1', 'two', '3']], dtype='<U11')
arr.shape #  (1, 3)

#1.random 난수 생성 함수 
np.random.randn(3,4) #모듈.모듈.함수()