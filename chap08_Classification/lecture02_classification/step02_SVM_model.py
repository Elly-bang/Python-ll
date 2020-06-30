# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:07:36 2020

SVM model

- 선형SVM, 비선형SVM
- Hyper parameter(kernel, C, gamma)
- 

"""

import pandas as pd #csv file read
from sklearn.model_selection import train_test_split #split
from sklearn.svm import SVC #model class 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score #model 평가 


#1.data set load
iris = pd.read_csv("C:/ITWILL/4_Python-ll/data/iris.csv")
iris.info()

 #2. x,y변수 선택 
 cols = list(iris.columns)
 cols
 
x_cols = cols[:4] #['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
y_col = cols[-1] #'Species'

#3. train(60)/test split(40)
train, test = train_test_split(iris,test_size = 0.4, random_state=123)

#4. model 생성
#1) 비선형 SVM model #kernel ='rbf' 
svc = SVC(C=1.0, gamma = 'auto', kernel ='rbf')
model = svc.fit(X=train[x_cols],y=train[y_col])
y_pred = model.predict(X=test[x_cols])
y_true = test[y_col]
acc = accuracy_score(y_true,y_pred)
acc # 0.9666666666666667 -> 0.95

#2) 선형 SVM model #kernel ='linear'
svc2 = SVC(C=1.0, gamma = 'auto', kernel ='linear') 
model2 = svc.fit(X=train[x_cols],y=train[y_col])
y_pred2 = model.predict(X=test[x_cols])
y_true2 = test[y_col]
acc2 = accuracy_score(y_true,y_pred)
acc2 #0.9666666666666667 -> 0.95


























