# -*- coding: utf-8 -*-
"""
step03_SMS_spam_classification.py

NB vs SVM : 희소행렬(고차원)
 - 가중치 적용 : Tfidf
"""

from sklearn.naive_bayes import MultinomialNB # NB model 
from sklearn.svm import SVC # SVM model 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가 
import numpy as np # npy file load # chap07/data -> spam_data.npy 

# 1. dataset load 
x_train, x_test, y_train, y_test=np.load('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/spam_data.npy', 
                                         allow_pickle=True)

x_train.shape # (3901, 4000)
x_test.shape # (1673, 4000)
# list -> numpy
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train.shape # (3901,)
y_test.shape # (1673,)


# 2. NB model
nb = MultinomialNB()
model = nb.fit(X = x_train, y = y_train)

y_pred = model.predict(X = x_test)
y_true = y_test


acc = accuracy_score(y_true, y_pred)
acc #  0.9736999402271369
con_mat = confusion_matrix(y_true, y_pred)
con_mat
'''
array([[1432,    0],
       [  44,  197]],
'''
print(197 / (44 + 197)) # 0.8174273858921162

# 3. SVM model 
svc = SVC(kernel='linear')
model_svc = svc.fit(X = x_train, y = y_train)

y_pred2 = model_svc.predict(X = x_test)
y_true2 = y_test

acc2 = accuracy_score(y_true2, y_pred2)
acc2 # 0.9784817692767483
con_mat2 = confusion_matrix(y_true2, y_pred2)
con_mat2
'''
array([[1427,    5],
       [  31,  210]],
'''
print(210 / (31+201)) # 0.9051724137931034









