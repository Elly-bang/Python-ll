# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:42:47 2020

교차검정 (CrossValidation)

"""
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate #cv 
from sklearn.metrics import accuracy_score

digit = load_digits()

X= digit. data
y= digit.target

X.shape #(1797, 64)
y #[0, 1, 2, ..., 8, 9, 8]

#2.model 
rf = RandomForestClassifier()
model = rf.fit(X,y)

pred = model.predict(X) #class 예측치  #[0, 1, 2, ..., 8, 9, 8]
pred2 = model.predict_proba(X) #확률 예측치 
pred2

'''
array([[0.98, 0.  , 0.  , ..., 0.  , 0.  , 0.01],
       [0.  , 0.99, 0.  , ..., 0.  , 0.  , 0.  ],
       [0.  , 0.03, 0.84, ..., 0.  , 0.12, 0.  ],
       ...,
       [0.02, 0.04, 0.  , ..., 0.  , 0.85, 0.  ],
       [0.  , 0.  , 0.02, ..., 0.  , 0.06, 0.9 ],
       [0.  , 0.01, 0.07, ..., 0.01, 0.83, 0.  ]])
'''
#확률 -> index(10진수)
pred2_dit = pred2.argmax(axis=1)
pred2_dit # [0, 1, 2, ..., 8, 9, 8]
acc = accuracy_score(y, pred)
acc #1.0

acc = accuracy_score(y, pred2_dit)
acc #1.0

# 3.CrossValidation
score = cross_validate(model, X, y , scoring='accuracy', cv=5)
score
'''
{'fit_time': array([0.21745634, 0.22006583, 0.22357297, 0.21866369, 0.20302033]),
 'score_time': array([0.        , 0.        , 0.        , 0.01565456, 0.        ]),
 'test_score': array([0.92777778, 0.91388889, 0.95264624, 0.96935933, 0.92479109])}
'''
test_score = score['test_score']

#산술 평균 
test_score.mean() #0.9376926648096564















