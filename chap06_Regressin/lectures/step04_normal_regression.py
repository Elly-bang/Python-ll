# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:06:06 2020

data scaling(정규화, 표준화) : 이물질 제거 
- 용도 : 특정 변수의 값에 따라서 모델에 영향을 미치는 경우
- ex) 범죄율(-0.1~0.99), 주택가격 (99~999)
- 정규화 : 변수의 값을 일정한 범위로 조정(min:0~max:1, -1~1) [X변수] 
           정규화 공식 nor = (x - min) / (max - min)
- 표준화 : 평균=0과 표준편차=1 이용 [Y변수]
           표준화 공식 z = ( x - mu ) / sd
"""
from sklearn.datasets import load_boston  #dataset 
from sklearn.linear_model import LinearRegression #model 생성
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import mean_squared_error, r2_score #model 평가

import numpy as np #min/max

#1. dataset load
X, y = load_boston(return_X_y=True)
X.shape
y.shape

#2. data scaling
'''
X 정규화 
Y 표준화 (평균=0, 표준편차=1)
'''

X.max() #711
X.mean() #70.07396704469443
y.max() #50
y.mean() #22.532806324110677

#정규화함수
def normal(x) :
    nor = (x - np.min(x)) / (np.max(x) -np.min(x))
    return nor 

#표준화함수 
def zscore(y):
    mu = y.mean()    
    z= (y - mu) / y.std()
    return z

X_nor = normal(X) 

# X변수 표준화 
x_nor = normal(X)
x_nor.mean() # 0.09855691567467571

# Y변수 표준화 (mu=0 , st=1)
y_nor = zscore(y)
y_nor.mean() #-5.195668225913776e-16
y_nor.std() #0.9999999999999999

#3.dataset split (75:25)
x_train, x_test, y_train, y_test =  train_test_split(x_nor, y_nor, random_state = 123) #test_size=0.25

x_train.shape  #(379, 13)
x_test.shape # (127, 13)

#4.model생성
lr = LinearRegression()
model = lr.fit(X=x_train, y=y_train)
model #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#5.model평가
y_pred = model.predict(X=x_test)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print('MSE=', mse) 
print('r2 score=', score)

# MSE= 0.2933980240643525 오류율 30%
# r2 score= 0.6862448857295749 정확률 70%










 