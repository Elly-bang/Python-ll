# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:36:01 2020

sklearn 관련 Linear Regression 

"""

from sklearn.model_selection import train_test_split #train/test split
from sklearn.metrics import mean_squared_error, r2_score #model 평가
from sklearn.linear_model import LinearRegression #model object
from sklearn.datasets import load_diabetes # dataset
import numpy as np #숫자 처리 
import pandas as pd  #상관계수
import matplotlib.pyplot as plt  #회귀선 시각화

##########################
######diabets
###########################
#단순선형회귀 : x(1) -> y

# 1.dataset load 
X, y= load_diabetes(return_X_y=True)
X.shape #(442, 10)
y.shape # (442,)
y.mean() #152.13348416289594

# 2. x,y 변수
# x(bmi:비만도 지수 )-> y

x_bmi = X[:,2]
x_bmi.shape #(442,)

# 1d -> 2d reshape
x_bmi = x_bmi.reshape(442, 1)

# 3. model생성 : object -> trainig -> model
obj = LinearRegression() #생성자 -> object 
model = obj.fit(x_bmi,y) #(X,y) -> model 

# y 예측치
#model.predict(x_bmi) #predict(X)
y_pred=  model.predict(x_bmi)
y_pred.shape #(442,)
y.shape #(442,)

# 4. model평가 : MSE(정규화), r2_score(비정규화)
MSE = mean_squared_error(y, y_pred) #(y 정답, y 예측치)
score = r2_score(y, y_pred)  #(y 정답, y 예측치)
print('mse=', MSE) # 3890.4565854612724
print('r2 score =', score) # 0.3439237602253803

# 5. dataset split(70:30)
x_train, x_test, y_train, y_test = train_test_split(x_bmi, y, test_size = 0.3)
x_train.shape  #(309, 1)
x_test.shape #(133, 1)

# model 생성
obj = LinearRegression() #생성자 -> object 
model = obj.fit(x_train, y_train) # traing dataset

y_pred = model.predict(x_test) #test dataset


# model평가 : MSE(정규화), r2_score(비정규화)
MSE = mean_squared_error(y_test, y_pred) #(y 정답, y 예측치)
score = r2_score(y_test, y_pred)  #(y 정답, y 예측치)
print('mse=', MSE) #3690.0621362483826
print('r2 score =', score) #0.32979400566699046

y_test[:10]
y_pred[:10]
y_test.mean()

import pandas as pd  #상관계수

df = pd.DataFrame({'y_true':y_test, 'y_pred':y_pred})
cor = df['y_true'].corr(df['y_pred'])
cor  #0.5854990714500304

import matplotlib.pyplot as plt #회귀선 시각화 
plt.plot(x_test, y_test, 'ro') #산점도
plt.plot(x_test, y_pred, 'b-') #회귀선
plt.show()

################
## iris.csv 
################
# 다중회귀모델 : y(1) <- x(2~4)

#1. dataset load
iris = pd.read_csv("C:/ITWILL/4_Python-ll/data/iris.csv")
iris.info()

'''
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Sepal.Length  150 non-null    float64 -> Y
 1   Sepal.Width   150 non-null    float64 -> x1
 2   Petal.Length  150 non-null    float64 -> x2
 3   Petal.Width   150 non-null    float64 -> x3
 4   Species       150 non-null    object 
 '''
 
#2. x,y 변수 선택
cols = list(iris.columns)
cols #['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

y_col = cols[0] #'Sepal.Length'
y_col 
x_cols = cols[1:-1] # ['Sepal.Width', 'Petal.Length', 'Petal.Width']
x_cols

# 3. dataset split(70:30)
iris_train, iris_test = train_test_split(iris, test_size=0.3, random_state=123)

#default = 0.25 

'''
test_size : 검정dataset 비율(default = 0.25)
random_state : sampling seed값 
'''
iris_train.shape #(105, 5)
iris_test.shape #(45, 5)

iris_train.head()
iris_test.head()

# 4. model 생성 : train data
lr = LinearRegression()
model = lr.fit(X=iris_train[x_cols],y=iris_train[y_col])
model #object info

# 5. model 평가 : test data
y_pred = model.predict(X=iris_test[x_cols]) #예측치
y_true = iris_test[y_col] #관측치(정답)=label

y_true.min() #4.3
y_true.max() #7.9

#평균제곱 오차 :mean((y_true - y_pred)**2)
mse = mean_squared_error(y_true, y_pred)
score = r2_score(y_true,y_pred)
#결정계수 : 1기준 
print('MSE=', mse) #mse= 0.11633863200224723
print('r2 score =', score) #r2 score =  0.8546807657451759(85%)

#y_ture vs y_pred 시각화 
'''
y_true[:10] #pandas.core.series.Series
y_pred[:10] #numpy.array
'''

#pandas -> Numpy 
y_true=np.array(y_true)
type(y_pred) #numpy.array
'''

import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(10, 5))
chart = fig.subplots()
chart.plot(y_true, color = 'b', label= 'fitted values')
chart.plot(y_pred, color = 'r', label= 'real values')
plt.legend(loc='best')
plt.show()







