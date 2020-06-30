# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:51:56 2020
@author: user

scipy 패키지의 stats 모듈의 함수
 - 통계적 방식의 회귀분석
 1.단순선형회귀모델 
 2.다중선형회귀모델

"""
from scipy import stats #회귀모델 
import pandas as pd #csv file read

#1.단순선형회귀모델
'''
x -> y
...

score_iq = pd.read_csv("C:/ITWILL/4_Python-ll/data/data/score_iq.csv")
score_iq.info()


#iq   독립변수로 잡아 성적과의 관계를 알아본다
#변수선택

x = score_iq.iq
y = score_iq['score']

#회귀모델 생성
model = stats.linregress(x, y)
model
'''
inregressResult
 slope=0.6514309527270075, 기울기 
 intercept=-2.8564471221974657, 절편
 rvalue=0.8822203446134699, 설명력 (1에 가까울 수록 높은 설명력)
 pvalue=2.8476895206683644e-50, x변수가 y에 미치는 영향, x의 유의성 검정
 stderr=0.028577934409305443 표본 오차 
'''
print('x=기울기', model.slope) #x=기울기 0.6514309527270075
print('y 절편 =', model.intercept) #y 절편 = -2.8564471221974657

#예측치 
score_iq.head(1)
'''
     sid  score   iq  academy  game  tv
0  10001     90  140        2     1   0
'''
#아이큐가 140일때
X=140 
# y = X * a + b 
X= 140
y_pred = X * model.slope + model.intercept 
y_pred #88.34388625958358

Y= 90
err = Y - y_pred
err #1.6561137404164157

##############
#product.csv 
#############
product = pd.read_csv("C:/ITWILL/4_Python-ll/data/data/product.csv")
product.info()


product. corr()

#x : 제품 적절성 -> y : 제품의 만족도 
model2 = stats.linregress(product['b'],product['c'])
model2

'''
LinregressResult(slope=0.7392761785971821, intercept=0.7788583344701907, rvalue=0.766852699640837, pvalue=2.235344857549548e-52, stderr=0.03822605528717565)
'''

product.head(1)
'''
   a  b  c
0  3  4  3
'''

X = 4
y_pred = X * model2.slope + model2.intercept
y_pred
Y =3
err = (Y-y_pred)**2
err #0.5416416092857157








