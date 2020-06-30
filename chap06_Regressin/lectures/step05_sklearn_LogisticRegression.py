# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:33:12 2020

sklearn 로지스틱 회귀모델
- y 변수가 범주형인 경우
"""

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression #model 생성
from sklearn.metrics import accuracy_score, confusion_matrix #model 평가 

#########################
#1. 이항분류 모델 
#########################

#1. dataset load  &변수 선택
breast = load_breast_cancer()

X = breast.data #x변수 
y = breast.target #y변수
y 
X.shape   #(569, 30)
y.shape   #(569,)

#2.model생성
help(LogisticRegression)

'''
random_state=None : 난수 seed값 지정 
solver='1bfgs' : 알고리즘 
  -solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, 
max_iter = 100 : 반복학습 횟수 
multi_class = 'auto' : 다항분류 
  -{'auto', 'ovr', 'multinomial'}, default='auto'
  
적용 예)
일반 데이터, 이항분류 : default
일반 데이터, 다항분류 : solver='1bfgs', multi_class ='multinomial'
빅 데이터 , 이항분류 : solver='sag' or 'saga'
빅 데이터 , 다항분류 : solver='sag' or 'saga', multi_class ='multinomial'
'''

lr = LogisticRegression(random_state=123)
model = lr.fit(X=X, y=y)
model #multi_class = 'auto' -> sigmoid 활용함수 ->  이항분류

# 3. model평가
acc = model.score(X, y)
print('accuracy =', acc) #accuracy = 0.9472759226713533 (95% 분류 정확도)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print('accuracy =', acc)  #accuracy = 0.9472759226713533 

con_max = confusion_matrix(y, y_pred)
print(con_max)
type(con_max) #numpy.ndarray
'''
   0    1
0[[193  19]
1[ 11 346]]
'''

acc = (con_max[0,0] + con_max[1,1]) / con_max.sum()
print('accuracy =', acc) #accuracy = 0.9472759226713533

import pandas as pd 

tab = pd.crosstab(y, y_pred, rownames=['관측치'], colnames=['예측치'])
tab 

'''
예측치      0    1
관측치          
0    193   19
1     11  346
'''
acc =( tab.loc[0,0] + tab.loc[1,1])/ len(y)
print('accuracy =', acc)   #accuracy = 0.9472759226713533



#########################
#2. 다항분류 모델 
#########################

# 1. dataset load
iris = load_iris()
iris.target_names #['setosa', 'versicolor', 'virginica'],

X, y = load_iris(return_X_y=True)

X.shape
y.shape
y #0~2

# 2. 모델 생성 
# 일반 데이터, 다항분류 : solver='1bfgs', multi_class ='multinomial'

lr = LogisticRegression(random_state= 123, solver='lbfgs', multi_class ='multinomial')

# multi_class ='multinomial' : softmax 활용함수 이용 -> 다항분류
'''
sigmoid function : 0~1 확률값 -> cutoff = 0.5 -> 이항분류
softmax function :  0~1 확률값 -> 전체합 = 1(c0:0.1, c1:0.1, c2:0.8) -> 다항분류 
'''
model = lr.fit(X, y)

y_pred = model.predict(X) #class 
y_pred2 = model.predict_proba(X) #확률값

y_pred   #0~2사이 
y_pred2.shape #벡터형식 (150, 3)
#['setosa', 'versicolor', 'virginica'] = 1
#[9.81797141e-01, 1.82028445e-02, 1.44269293e-08],

import numpy as np
arr = np.array([9.81797141e-01, 1.82028445e-02, 1.44269293e-08])
arr.max() #0.981797141
arr.min() #1.44269293e-08
arr.sum() # 0.9999999999269293

#3.모델 평가 
acc = accuracy_score(y, y_pred)
print('accuracy =', acc) #0.9733333333333334

con_max = confusion_matrix(y, y_pred)
con_max
'''array([[50,  0,  0],
       [ 0, 47,  3],
       [ 0,  1, 49]],
'''

acc =( con_max[0,0]+ con_max[1,1]  + con_max[2,2] ) /  con_max.sum()
print('accuracy =', acc) #accuracy = 0.9733333333333334


#히트맵 : 시각화 
import seaborn as sn 
import matplotlib.pyplot as plt

#confusion matrix heatmap
plt.figure(figsize=(6,6)) #chart size
sn.heatmap(con_max, annot=True, fmt=".3f", linewidth=.5, square = True)
plt.ylabel('Actual label')
plt.xlabel('Predict label')
all_sample_title = 'Accuracy Score:{0}'.format(acc)
plt.title(all_sample_title, size =18)
plt.show()

