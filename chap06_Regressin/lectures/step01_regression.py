# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:44:25 2020

회귀방정식에서 기울기 (slope)와 절편(intercept)식
기울기(slope) = Cov(x,y) / Sxx(x편차의 제곱의 평균)
절편(intercept) = y_mu - ( slope * x_mu )
"""

from scipy import stats #회귀모델 
import pandas as pd #csv file read

galton=pd.read_csv('C:/ITWILL/4_Python-ll/data/galton.csv')
galton.info()
'''
class 'pandas.core.frame.DataFrame'>
RangeIndex: 928 entries, 0 to 927
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   child   928 non-null    float64
 1   parent  928 non-null    float64
'''
 
#x,y변수 선택
 x=galton['parent']
 y=galton['child']
 
 #model 생성 
 model = stats.linregress(x,y)
 model
'''
LinregressResult(slope=0.6462905819936423,
 intercept=23.941530180412748, 
 rvalue=0.4587623682928238, (상관관계가 높지 않다. 부모의 키가 자녀의 키에 100% 반영은 아님) 
 pvalue=1.7325092920142867e-49,
 stderr=0.04113588223793335)
'''

# 회귀방정식 :  Y = x * a + b 
y_pred = x * model.slope + model.intercept
y_pred

y_true = y 

# 예측치 vs 관측치 (정답)
y_pred.mean() #68.08846982758534
y_true.mean() #68.08846982758512

#기울기 계산식
#기울기(slope) = Cov(x,y) / Sxx(x편차의 제곱)
xu = x.mean() #x_mu
yu = y.mean() #y_mu

Cov_xy = sum ((x- xu) * ( y-yu ))/ len(x)
Cov_xy #2.062389686756837

Sxx = np.mean((x- xu) **2)
Sxx  #3.1911182743757336


slope = Cov_xy /Sxx
slope # 0.6462905819936413

#2. 절편 계산식 
#절편(intercept) = y_mu - ( slope * x_mu )

intercept = yu - (slope * xu )
intercept #23.94153018041171

#3. rvlaue 설명력 
galton.corr()

'''
        child    parent
child   1.000000  0.458762
parent  0.458762  1.000000
'''
y_pred =  x * slope + intercept
y_pred.mean() #68.08846982758423


































































































































































