# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:54:01 2020

집단 간 평균차이 검정(t_test)
1. 한 집단 평균 차이 검정
2. 두 집단 평균 차이 검정
3. 대응 두 집단 평균 차이 검정
 
"""
from scipy import stats # t검정
import numpy as np # 숫자 연산
import pandas as pd #file read

# 1. 한 집단 평균 차이 검정

# 대한민국 남자 평ㅇ균 키 (모평균) : 175.5cm 
# 모집단 -> 표준추출(30명)

sample_data = np.random.uniform(172,180,size=300)
sample_data

#기술통계
sample_data.mean() #176.04836652153554

one_group_test = stats.ttest_1samp(sample_data,175.5)
one_group_test #statistic=3.989340324697975, pvalue=8.338525114502645e-05
print('statistic= %.5f, pvalue=%.5f'%(one_group_test))
#statistic= 3.98934, pvalue=0.00008 < 0.05

#2. 두 집단 평균 차이 검정

female_score = np.random.uniform(50, 100,size= 30)
male_score = np.random.uniform(45, 95, size= 30)
two_sample = stats.ttest_ind(female_score, male_score)
two_sample #statistic=1.0228788244611733, pvalue=0.3106126677353999 >= 0.05 
print('statistic= %.5f, pvalue=%.5f'%(two_sample))
#statistic= 1.02288, pvalue=0.31061

#기술통계 
female_score.mean() #77.15114832469574
male_score.mean() # 73.06936176528616

#csv file load
two_sample = pd.read_csv('C:/ITWILL/4_Python-ll/data/data/two_sample.csv')
two_sample.info()


      no  gender  method  score  survey
0      1       1       1    5.1       1
1      2       1       1    5.2       0
2      3       1       1    4.7       1
3      4       2       1    NaN       0
4      5       1       1    5.0       1
..   ...     ...     ...    ...     ...
235  236       2       2    NaN       1
236  237       1       2    5.4       1
237  238       2       2    6.0       1
238  239       2       2    6.7       1
239  240       2       2    5.2       0

sample_data = two_sample[['method', 'score']]
sample_data.head()
'''
   method  score
0       1    5.1
1       1    5.2
2       1    4.7
3       1    NaN
4       1    5.0
'''
sample_data['method'].value_counts()

'''
2    120
1    120
'''
#교육방법에 따른.subset
method1 = sample_data [sample_data ['method']== 1]
method2 = sample_data [sample_data ['method']== 2]
score1 = method1.score
score2 = method2.score

#NA -> 평균 대체 
score1 = score1.fillna(score1.mean())
score2 = score2.fillna(score2.mean())

#na-> error
two_sample = stats.ttest_ind(score1,score2)
two_sample
print('statistic= %.5f, pvalue=%.5f'%(two_sample))
#statistic= -0.94686, pvalue=0.34467

score1.mean() # 5.496590909090908
score2.mean() #5.496590909090908

#3.대응 두 집단 평균 차이 검정: 복용전 65 -> 복용후 :60 변환
before = np.random.randint(65, size = 30 ) *0.5
after = np.random.randint(60, size= 30) *0.5

before.mean() #14.15
after.mean()  #12.6

paired_test = stats.ttest_rel(before, after)
paired_test #statistic=0.6512177564676991, pvalue=0.5200333353613923
print('statistic= %.5f, pvalue=%.5f'%(paired_test))
#(paired_test))statistic= 0.65122, pvalue=0.52003




