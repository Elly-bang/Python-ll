# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:48:30 2020
NB 모델
GaussianNB :X 변수가 실수형이고, 정규분포 형태
MultinomialNB : 희소행렬과 같은 고차원 데이터 이용하여 다항분류에 이용 

"""
import pandas as pd #csv file read
from sklearn.model_selection import train_test_split #split
from sklearn.naive_bayes import GaussianNB, MultinomialNB #model class 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score #model 평가 
from scipy import stats #정규분포 검정 
 
#######################################
### GaussianNB 
#######################################

#1.data set load
iris = pd.read_csv("C:/ITWILL/4_Python-ll/data/iris.csv")
iris.info()
'''
 0   Sepal.Length  150 non-null    float64
 1   Sepal.Width   150 non-null    float64
 2   Petal.Length  150 non-null    float64
 3   Petal.Width   150 non-null    float64
 4   Species       150 non-null    object 
 '''
 
 #정규성 검정
 stats.shapiro(iris['Sepal.Width']) #Sepal.Width   0.10113201290369034 정규분포
 
 #2. x,y변수 선택 
 cols = list(iris.columns)
 cols
 
x_cols = cols[:4] #['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
y_cols = cols[-1] #'Species'

#3. train/test split
train, test = train_test_split(iris,test_size = 0.3, random_state=123)

#4.NB model
nb = GaussianNB()
model = nb.fit(X=train[x_cols], y=train[y_cols])
model  #GaussianNB(priors=None, var_smoothing=1e-09)

#5. model 평가 
y_pred = model.predict(X=test[x_cols])
y_true = test[y_cols]


acc = accuracy_score(y_true, y_pred) #분류정확도
con_mat = confusion_matrix(y_true, y_pred) #교차분할표
f1_score = f1_score(y_true, y_pred, average='micro') #불균형인 경우 

acc # 0.9555555555555556
con_mat
'''
array([[18,  0,  0],
       [ 0, 10,  0],
       [ 0,  2, 15]], dtype=int64)

'''
f1_score #정확률 재현률 -> 조화평균 #0.9555555555555556


###################
##MultinomialNB
###################

1. dataset load 
from sklearn.datasets import fetch_20newsgroups
newsgroups= fetch_20newsgroups(subset='all') # 'train', 'test'
# Downloading 20news dataset.
print(newsgroups.DESCR)

'''
x변수 : news기사 내용 (text자)
y변수 : 해당 news group(20)
'''
newsgroups.target_names
cats = newsgroups.target_names[:4]
cats #['alt.atheism','comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware']

#2.text -> sparse matrix (subset = 'train')

from sklearn.feature_extraction.text import TfidfVectorizer

news_train= fetch_20newsgroups(subset='train', categories= cats) 
news_train.data  #text : x변수
news_train.target #y변수 [3, 2, 2, ..., 0, 1, 1]

#3. sparse matrix
tfidf = TfidfVectorizer()
sparse_mat = tfidf.fit_transform(news_train.data )
sparse_mat.shape #(2245, 62227)


#3. model
nb= MultinomialNB()
model =  nb.fit(X=sparse_mat,y=news_train.target) 

#4. model 평가 fetch_20newsgroups(subset = 'test')
news_test = fetch_20newsgroups(subset = 'test', categories=cats)
news_test.data   #text 
news_test.target  #[1, 1, 1, ..., 1, 3, 3]


sparse_test = tfidf.transform(news_test.data)
sparse_test.shape

'''
obj.fit_transform(train_data)
obj.transform(test_data)
'''

y_pred = model.predict(X=sparse_test)
y_pred.shape
y_true= news_test.target 
y_true.shape


acc = accuracy_score(y_true, y_pred) #분류정확도
con_mat = confusion_matrix(y_true, y_pred) #교차분할표
f1_score = f1_score(y_true, y_pred, average='micro') #불균형인 경우 

acc 
con_mat
acc2 = (con_mat[0,0]+con_mat[1,1]+con_mat[2,2]+con_mat[3,3])/con_mat.sum()
acc2 
f1_score 










