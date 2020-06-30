# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:07:19 2020
@author: user

1. image shape : 3차원의 구조 (세로, 가로, 칼)
2. reshape : 사이즈 변경 안됨 
    ex) [2, 5] -> [5, 2] o
        [3, 4] -> [4, 2] x
"""


import numpy as np
from matplotlib.image import imread # image읽기
import matplotlib.pylab as plt 

import numpy as np
from matplotlib.image import imread
import matplotlib.pylab as plt

# 1.image shape
file_path='C:/ITWILL/4_Python-ll/workspace/chap04_NumPy/images/test1.jpg'
image=imread(file_path)
type(image) # numpy.ndarray

image.shape #(360, 540, 3) ->(가로, 세로, 컬러)
print(image)
plt.imshow(image)

#1. image shape
file_path='C:/ITWILL/4_Python-ll/workspace/chap04_NumPy/images/test1.jpg'
image=imread(file_path)
type(image)

image.shape
print(image)
plt.imshow(image)

#RGB색상 분류 
r= image[:,:,0] #R
g= image[:,:,1] #G
b= image[:,:,2] #B
r.shape # (360, 540)
g.shape 
b.shape 

#2. image data reshape
from sklearn.datasets import load_digits #데이터셋 제공

digit = load_digits() #dataset loading
digit.DESCR #설명보기 

X = digit.data #x변수 (입력변수) : image 
y = digit.target #y변수(정답 = 정수)
X.shape #(1797, 64) 64=8X8 
y.shape #(1797,)

img_0 = X[0].reshape(8,8) #행 index
img_0
plt.imshow(img_0) #0 
y[0] #0

img_0=X[3].reshape(8,8)
plt.imshow(img_0)  


X_3d = X.reshape(-1,8,8) 
X_3d.shape #(1797, 8, 8) -> (전체이미지, 세로, 가로, [칼럼])

#(1797, 8, 8, 1)
X_4d = X_3d[:, :, :,np.newaxis] #4번축 추가 
X_4d.shape #(1797, 8, 8, 1)

#3. reshape
'''
전치행렬 : T 
swapaxis = 전치행렬
transpose() : 3차원이상 모양 변경 
'''
#1)전치행렬
data = np.arange(10).reshape(2,5)
\]
data
'''
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
'''
print(data.T)
'''
[[0 5]
 [1 6]
 [2 7]
 [3 8]
 [4 9]]
'''

#2) transpose()

'''
1차원 : 효과 없음
2차원 : 전치행렬 동일함
3차원 : (0.1.2) -> (2,1,0)
'''

arr3d = np.arange(1,25).reshape(4,2,3)
arr3d.shape # (4, 2, 3)
arr3d 

#(0,1,2) -> (2,1,0)
arr3d_tran=arr3d.transpose(2,1,0)
arr3d_tran.shape #(3,2,4)

#(0,1,2) -> (1,2,0)
arr3d_tran=arr.3d.transpose(1,2,0)

arr3d_tran.shape #(3, 2, 4)

#2. image data reshape
from sklearn.datasets import load_digits

digit = load_digits()
digit.DESCR

x=digit.data
y=digit.target
x.shape
y.shape


img_0 = x[0].reshape(8,8)
img_0 






