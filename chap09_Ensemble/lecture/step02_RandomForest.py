# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:42:26 2020

RandomForest
"""

from sklearn.ensemble import RandomForestClassifier #model
from sklearn.datasets import load_wine #dataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

#1.dataset
wine = load_wine()
wine.feature_names # x변수명 
wine.target_names #y범주 이름 : ['class_0', 'class_1', 'class_2']

X = wine.data 
y = wine.target

X.shape #(178, 13)

#2. RF  model

rf = RandomForestClassifier()

'''
n_estimators : integer, optional (default=100) : 트리수 
criterion : string, optional (default="gini") : 중요변수 선정 기준
max_depth : integer or None, optional (default=None) : 트리 깊이
min_samples_split : int, float, optional (default=2) : 노드 분할 최소 샘플수
min_samples_leaf : int, float, optional (default=1) : 단노드 분할 최소수 
max_features : int, float, string or None, optional (default="auto") : 최대 X 변수 사용수 
random_state : int, RandomState instance or None, optional (default=None) : 시드 값 지정
n_jobs : int or None, optional (default=None) : cpu 수  
'''
rf 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score 
import numpy as np
idx = np.random.choice(a=X.shape[0], size=int(X.shape[0]*0.7), replace= False)

X_train = X[idx] #X[idx, :]
y_train = y[idx]

model = rf.fit(X=X_train,y=y_train)

idx_test = [i for i in range(len(X))if not i in idx]
len(idx_test) #54
x_test = X[idx_test]
y_test = y[idx_test]

x_test.shape #(54, 13)

y_pred = model.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true,y_pred)
report = classification_report(y_true, y_pred)
con_mat = confusion_matrix(y_true, y_pred)

report = classification_report(y_true, y_pred)

print('중요변수 :' , model.feature_importances_)

# [0.09412805 0.03204756 0.00785571 0.0386894  0.02405025 0.06415233
 0.18256316 0.01409003 0.02950838 0.16004712 0.05809446 0.09814714
 0.19662642]

len(model.feature_importances_) #13

#중요변수 시각화
import matplotlib.pyplot as plt

x_size = X.shape[1]
plt.barh(range(x_size), model.feature_importances_) #(y,x)
plt.yticks(range(x_size),wine.feature_names)
plt.xlabel('importance')
plt.show()










