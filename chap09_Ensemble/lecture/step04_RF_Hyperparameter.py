# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:34:44 2020
RandomForest Hyper Parameter
step02_RandomForest참고
"""

from sklearn.ensemble import RandomForestClassifier #model
from sklearn.datasets import load_wine #dataset
from sklearn.metrics import confusion_matrix


#1.dataset
wine = load_wine()
wine.feature_names # x변수명 
wine.target_names #y범주 이름 : ['class_0', 'class_1', 'class_2']

X = wine.data 
y = wine.target

#2. RF  model

rf = RandomForestClassifier()sdrg


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
model = rf.fit(X=X,y=y)

#3.Grid Search 

from sklearn.model_selection import GridSearchCV

#1) params설정 
params = [{'n_estimators':[100,200,300,400],
           'max_depth' : [3,6,8,10],
           'min_samples_split':[2,3,4,5],
           'min_samples_leaf':[1,3,5,7]}]

             
#2) GridSearchCV 객체 
gs = GridSearchCV(estimator = model, param_grid = params, scoring='accuracy',cv=10, n_jobs=-1) 

#scoring : 평가방법 cv:교차검정 n_jobs : cpu수
model = gs.fit(X, y)

#best score
acc = model.score(X, y)
acc #0.9943820224719101

#best parameter
model.best_params_
'''
{'max_depth': 3,
 'min_samples_leaf': 1,
 'min_samples_split': 3,
 'n_estimators': 400}
'''


























