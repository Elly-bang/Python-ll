# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:35:01 2020

행렬곱 함수 (np.dot) 이용 y예측치 구하기 
y_pred = np.dot(X,a) + b 

np.dot(X, a) 전제조건
1. X, a : 행렬 구조
2. 수일치 : X열 차수 = a의 행 차수 

"""
from scipy import stats #단순회귀모델 
from statsmodels.formula.api import ols # 다중회귀모델 
import pandas as pd #file read
import numpy as np #list -> numpy


#1. dataset load
score = pd.read_csv("C:/ITWILL/4_Python-ll/data/score_iq.csv") 
score.info()

'''
 0   sid      150 non-null    int64
 1   score    150 non-null    int64 -> y
 2   iq       150 non-null    int64 -> X1
 3   academy  150 non-null    int64 -> X2
 4   game     150 non-null    int64
 5   tv       150 non-null    int64
 '''
 
 
formula = "score ~ iq + academy"
model = ols(formula, data = score).fit()

#회귀계수 : 기울기, 절편
model.params
'''
Intercept    25.229141
iq            0.376966
academy       2.992800
'''

#model 결과 확인 
model.summary()
"""

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  score   R-squared:                       0.946
Model:                            OLS   Adj. R-squared:                  0.946
Method:                 Least Squares   F-statistic:                     1295.
Date:                Tue, 12 May 2020   Prob (F-statistic):           4.50e-94
Time:                        11:45:22   Log-Likelihood:                -275.05
No. Observations:                 150   AIC:                             556.1
Df Residuals:                     147   BIC:                             565.1
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     25.2291      2.187     11.537      0.000      20.907      29.551
iq             0.3770      0.019     19.786      0.000       0.339       0.415
academy        2.9928      0.140     21.444      0.000       2.717       3.269
==============================================================================
Omnibus:                       36.342   Durbin-Watson:                   1.913
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               54.697
Skew:                           1.286   Prob(JB):                     1.33e-12
Kurtosis:                       4.461   Cond. No.                     2.18e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.18e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

#model예측치
model.fittedvalues

#y_pred = np.dot(X,a)+b 

X=score[['iq','academy']]
X.shape #(150, 2)  # x1, x2 

'''
np.dot(X,a) 전제조건
1. X, a : 행렬 구조
2. 수일치 : X열 차수 = a의 행 차수 
'''

#list -> numpy
a = np.array([[0.376966],[2.992800]]) #(2,1)
a.shape #(2,1)

matmul = np.dot(X,a) #행렬곱
matmul.shape #(150, 1) = X(150,2).a(2,1)

b =  25.229141 #절편
y_pred =  matmul +  b #broadcast (2차원 + 0차원)
y_pred.shape  #(150, 1)

#2차원 -> 1차원 : reduce
y_pred1d = y_pred.reshape(150)
y_pred1d.shape #(150,)

y_true = score['score']
y_true.shape  #(150,)

df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred1d})
df.head()
df.tail()

cor = df['y_true'].corr(df['y_pred'])
cor 


























 









