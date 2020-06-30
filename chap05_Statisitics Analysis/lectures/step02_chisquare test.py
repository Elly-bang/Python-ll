# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:39:36 2020

@author: user

카이제곱검정(chisquare test)
 - 일원 카이제곱, 이원 카이제곱
"""

from scipy import stats
import numpy as np


# 1. 일원 카이제곱 검정 
# 귀무가설 : 관측치와 기대치는 차이가 없다.(게임에 적합하다.) 
# 대립가설 : 관측치와 기대치는 차이가 있다.(게임에 적합하지 않다.) 
real_data = [4, 6, 17, 16, 8, 9] # 관측치 
exp_data = [10,10,10,10,10,10] # 기대치 
chis = stats.chisquare(real_data, exp_data) 
chis
print('statistic = %.3f, pvalue= %.3f'%(chis)) 
# statistic = 14.200, pvalue = 0.014

# 2.이원 카이제곱 검정 
import pandas as pd
smoke = pd.read_csv('data/smoke.csv')
smoke.info() # 355x2

# DF -> vector
education = smoke.education
smoking = smoke.smoking

# cross table
table = pd.crosstab(education, smoking)
table
'''
smoking     1   2   3
education            
1          51  92  68
2          22  21   9
3          43  28  21
'''

chis = stats.chisquare(education, smoking)
chis

# statistic=347.66666666666663, pvalue=0.5848667941187113
# pvalue=0.5848667941187113 >= 0.05 : 귀무가설 채택
# [해석] 교육수준과 흡연유무에는 관계가 없다

'''
성별 vs 흡연유무 독립성 검정
'''
tips = pd.read_csv('data/tips.csv')
tips.info()
table = pd.crosstab(tips.sex, tips.smoker)
table
'''
smoker  No  Yes
sex            
Female  54   33
Male    97   60
'''
gender = tips.sex
smoker = tips.smoker

gender_dummy = [1 if g =='Male' else 2 for g in gender]
smoker_dummy = [1 if s =='No' else 2 for s in smoker]
pd.crosstab(gender_dummy, smoker_dummy)



chis = stats.chisquare(gender_dummy, smoker_dummy)
chis # statistic=84.0, pvalue=1.0



import pandas as pd 
smoke = pd.read_csv('data/data/smoke.csv')
smoke.info()

#DF -> vextor

education= smoke.education
smoking = smoke.smoking



