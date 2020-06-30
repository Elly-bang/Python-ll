# -*- coding: utf-8 -*-
"""
문) california 주택가격을 대상으로 다음과 같은 단계별로 선형회귀분석을 수행하시오.
"""

# california 주택가격 데이터셋 
'''
캘리포니아 주택 가격 데이터(회귀 분석용 예제 데이터)

•타겟 변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) 
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도
'''

from sklearn.datasets import fetch_california_housing # dataset load
import pandas as pd # DataFrame 생성 
from sklearn.linear_model import LinearRegression  # model
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import mean_squared_error, r2_score # model 평가 
import matplotlib.pyplot as plt 

# 캘리포니아 주택 가격 dataset load 
california = fetch_california_housing()
print(california.DESCR)

california.feature_names
california.target


# 단계1 : 특징변수와 타켓변수(MEDV)를 이용하여 DataFrame 생성하기   
cal_df = pd.DataFrame(california.data, columns= california.feature_names)
cal_df["MEDV"]= california.target
cal_df.info()
cal_df.tail()
cal_df.shape

            
# 단계2 : 타켓변수와 가장 상관관계가 높은 특징변수 확인하기  
cor = cal_df.corr()
cor
cor.MEDV
'''
cor.MEDV
Out[136]: 
MedInc        0.688075   !!
HouseAge      0.105623
AveRooms      0.151948
AveBedrms    -0.046701
Population   -0.024650
AveOccup     -0.023737
Latitude     -0.144160
Longitude    -0.045967
MEDV          1.000000
Name: MEDV, dtype: float64
'''


# 단계3 : california 데이터셋을 대상으로 1만개 샘플링하여 서브셋 생성하기  
import numpy as np
idx = np.random.choice(a=len(cal_df), size = 10000, replace=False)
len(idx) # 10000
idx

#행번호 이용  10000개 샘플링
cal_data = cal_df.loc[idx]
cal_data.shape
cal_data.info()
cal_data



# 단계4 : 단계3에서 만든 서브셋을 대상으로 75%(train) vs 25(test) 비율 데이터셋 split 
train, test = train_test_split(cal_data, random_state=123)
train.shape
test.shape



# 단계5 : 선형회귀모델 생성

lr = LinearRegression()
model = lr.fit(train.iloc[:, :8], train.iloc[:,8])  # fit(x, y)


# 단계6 : 모델 검정(evaluation)  : 예측력 검정, 과적합(overfitting) 확인  
train_acc = model.score(train.iloc[:, :8],train.iloc[:,8])
test_acc = model.score(test.iloc[:, :8],test.iloc[:,8])

print('train accuracy =', train_acc) #train accuracy = 0.6198307237250431
print('test accuracy =', test_acc) # test accuracy = 0.5951110878530291

'''
train_acc
test_acc
[해설] 훈련셋과 검정셋 모두 비슷한 분류 정확도 -> 과적합 없음 
'''


# 단계7 : 모델 평가(test) 
# 조건1) 단계3의 서브셋 대상으로 30% 샘플링 자료 이용

test_idx = np.random.choice(a=len(cal_data), size = int(len(cal_data)*0.3), replace=False)
len(test_idx) #3000

test_data=cal_df.iloc[test_idx, :]
test_data.shape # (3000, 9)
y_pred = model.predict(test_data.iloc[:, :8])
y_true = test_data.iloc[:,8] #관측치 

# 조건2) 평가방법 : MSE, r2_score
mse = mean_squared_error(y_true, y_pred)
score = r2_score(y_true,y_pred)

print('MSE = ', mse) #MSE =  0.5772783722142919
print('r2_score= ', score) #r2_score=  0.5777858204322438


type(y_pred) #np.ndarray
type(y_true) #pandas.core.series.Series
y_true = np.array(y_true)

# 단계8 : 예측치 100개 vs 정답 100개 비교 시각화 
plt.plot(y_true[:100], color='b', label='real values')
plt.plot(y_pred[:100], color='r', label='fitted values')
plt.xlabel('index')
plt.ylabel('fitted values')
plt.legend(loc = 'best')
plt.show()


import seaborn as sn 
import matplotlib.pyplot as plt

#confusion matrix heatmap
plt.figure(figsize=(6,6)) #chart size
sn.heatmap(con_max, annot=True, fmt=".3f", linewidth=.5, square = True)
plt.ylabel('Actual label')
plt.xlabel('Predict label')
all_sample_title = 'Accuracy Score:{0}'.format(acc)
plt.title(all_sample_title, size =18)
plt.show()













