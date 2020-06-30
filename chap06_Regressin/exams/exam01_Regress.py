'''
문) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성 
  조건1> train/test - 7:3비울
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> 모델 평가 : MSE, r2_score
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

# 1. data load
boston = load_boston()
print(boston)


# 2. 변수 선택  
y = boston.target
x = boston.data

# 3. train/test split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# 4. 회귀모델 생성 : train set
lr= LinearRegression()
model = lr.fit(x_train,y= y_train)
model #LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# 5. 모델 평가 : test set
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print('MSE =', mse)
print('r2 score =' , score)

type(y_pred)
type(y_test)

#시각화
import matplotlib.pyplot as plt 
fig  = plt.figure(figsize=(10,4))
chart = fig.add_subplot()
chart.plot(y_test, color='b', label = 'real values')
chart.plot(y_pred, color='r', label = 'fitted values')
plt.title('real values vs fitted values')
plt.xlabel('index')
plt.ylabel('prediction')
plt.xlabel(loc ='best')
plt.show()
