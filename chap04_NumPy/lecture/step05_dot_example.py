# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:54:11 2020
@author: user

신경망 행렬곱 적용 예 
 - 은닉층(h) = [입력(X)*가중치(w)] + 편향(b)
"""

import numpy as np

#1. ANN model
#input : image(28x28), hidden node: 32개  -> weight[?,?]

#2. input data : image data
28 *28  #784
X_img = np.random.randint(0, 256, size = 784)
X_img.shape #(784,)
X_img.max() #255 0부터 255까지 


#이미지 정규화 : 0~1
X_img = X_img / 255
X_img
X_img.max() #1.0
X_img2d=X_img.reshape(28,28) 
X_img2d
X_img2d.shape #(28, 28)

#3.weight data
weight=np.random.randn(28, 32)
weight
weight.shape # (28, 32)

#4.hidden layer
hidden = np.dot(X_img2d, weight)
hidden.shape # h(28, 32) = x(28, 28) * w(28, 32)
