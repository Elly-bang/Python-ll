# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:07:17 2020

Pipeline vs Grid Search
1. SVM model
2. Pipeline : model workflow (dataset 전처리 -> model -> test)
3. Grid Search : model tuning

"""

from sklearn.datasets import load_breast_cancer #datase
from sklearn.svm import SVC  #model_class
from sklearn.model_selection import train_test_split #split 
from sklearn.preprocessing import MinMaxScaler #scaling(0~1)
from sklearn.pipeline import Pipeline #model workflow
import numpy as np

#1. SVM model

# 1) data load 
X, y =load_breast_cancer(return_X_y=True)
X.shape  #(569, 30)

# 열 평균 
X.mean(axis=0)
# 1.41272917e+01 
# 6.54889104e+02,
X.min() #0.0
X.max() #4254.0


# 2) X변수 정규화 : 전처리
scaler = MinMaxScaler().fit(X) #scaler객체
X_nor = scaler.transform(X) # 정규화
X_nor.mean(axis=0)
X_nor.min() #0.0 
X_nor.max() #1.0000000000000002

x_train, x_test, y_train, y_test = train_test_split(X_nor, y, test_size=0.3)

# 3) SVM model 생성 
svc = SVC(gamma='auto') # 비선형 SVM model 
model = svc.fit(x_train, y_train)

# 4) model평가
score = model.score(x_test, y_test)
score # 0.9649122807017544

#2. Pipeline

# 1) pipeline step : [ (step1:scaler),(step2:model),....]
pipe_svc = Pipeline([('scaler',MinMaxScaler()),('svc',SVC(gamma='auto'))])

# 2) pipeline model 
model = pipe_svc.fit(x_train, y_train)

# 3) pipeline model test
score = model.score(x_test, y_test)       


#3. Grid Search : model tuning
#Pipeline -> Grid Search -> modle tuning
from sklearn.model_selection import GridSearchCV

#help(SVC) 
# C=1.0, kernel='rbf', degree=3, gamma='auto'

#1) params설정 
params =[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#dict 형식= {'object__C': params_range) 
params_grid = [{'svc__C':params,'svc__kernel':['linear']}, #선형 
               {'svc__C':params,'svc__gamma':params,'svc__kernel':['rbf']}] #비선형
             
#2) GridSearchCV 객체 
gs = GridSearchCV(estimator = pipe_svc, param_grid = params, scoring='accuracy',cv=10, n_jobs=1) 
#scoring : 평가방법 cv:교차검정 n_jobs : cpu수
model = gs.fit(x_train, y_train)

#best score
acc = model.score(x_test, y_test)
acc # 0.9532163742690059


#best parameter
model.best_params_


















                                                                 g
