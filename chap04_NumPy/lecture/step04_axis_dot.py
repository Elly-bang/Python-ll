# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:49:08 2020
@author: user
1. 축 (axis) :행축, 열축
2. 행렬곱 연산 : np.dot()
    회귀방정식 행렬곱 예= [X * a] + b 
    X1, X2 -> a1, a2
    model = [X1 * a1 + X2 * a2 ] +b
    model = np.dot(X, a) +b 
    
    -신경망에서 행렬곱 예 
    [X * w] + b 
    
"""

import numpy as np

#1. 축(axis)
'''
행축 : 동일한 열의 모음 (axis=0) -> 열단위
열축 : 동일한 행의 모음 (axis=1) -> 행단위
'''

arr2d = np.random.randn(5,4)
arr2d
print('전체 원소 합계 : ', arr2d.sum()) #전체 원소 합계 :  -2.1103620196108945
print('행 단위 합계 : ', arr2d.sum(axis=1))
#행 단위 합계 :  [-1.70620055  0.28922357 -0.43258538  1.12215395 -1.38295361]
print('열 단위 합계 : ', arr2d.sum(axis=0))
#열 단위 합계 :  [-1.01305399 -1.77362706 -1.88598165  2.56230068]

#2. 행렬곱 연산 : np.dot()
X = np.array([[2,3],[2.5,3]])
X #입력 x
'''
array([[2. , 3. ],
       [2.5, 3. ]])
'''

X.shape #(2,2)

a = np.array([[0.1],[0.05]])
a
a.shape #(2, 1)

#b=0.1 절편
b=0.1
y_pred= np.dot(X,a) #+b
'''
np.dot(X,a)전제조건
1. X, a : 행렬구조
2. 수일치 : X열 차수 = a의 행차수 
'''

y_pred 
'''
array([[0.35],
       [0.4 ]])
'''

#[실습] p.60
X = np.array([[0.1, 0.2], [0.3, 0.4]])
X.shape  #(2, 2)
X

'''
array([[0.1, 0.2],
       [0.3, 0.4]])
'''

W=np.array([[1,2,3],[2,3,4]])
W
W.shape # (2, 3)

#행렬곱
h = np.dot(X,W)
h
'''
array([[0.5, 0.8, 1.1],
       [1.1, 1.8, 2.5]])
'''

h.shape #h(2, 3) = X(2,2) * W(2,3)

































