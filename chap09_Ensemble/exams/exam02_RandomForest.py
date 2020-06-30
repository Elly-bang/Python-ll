'''
 문) 당뇨병(diabetes.csv) 데이터 셋을 이용하여 다음과 같은 단계로 
     RandomForest 모델을 생성하시오.

  <단계1> 데이터셋 로드
  <단계2> x,y 변수 생성 : y변수 : 9번째 칼럼, x변수 : 1 ~ 8번째 칼럼
  <단계3> 500개의 트리를 random으로 생성하여 모델 생성 
  <단계4> 5겹 교차검정/평균 분류정확도 출력
  <단계5> 중요변수 시각화 
'''

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt # 중요변수 시각화 

# 단계1. 테이터셋 로드  
dia = pd.read_csv('../data/diabetes.csv', header=None) # 제목 없음  
print(dia.info()) 
print(dia.head()) 

# 단계2. x,y 변수 생성 

X = dia.loc[:, 0:7]
y = dia.loc[:, 8]
X.shape #(759,8)
y.shape #(759, )

feature_names = dia.columns
feature_names

# 단계3. model 생성
rf = RandomForestClassifier()
model = rf.fit(X,y)

# 단계4. 교차검정 model 예측/평가 
score = model_selection.cross_validate(model, X, y, cv=5,)
score 
'''
{'fit_time': array([0.1103189 , 0.10928988, 0.12502694, 0.12982416, 0.12502813]),
 'score_time': array([0.01561832, 0.        , 0.        , 0.        , 0.        ]),
 'test_score': array([0.73684211, 0.73684211, 0.75      , 0.80921053, 0.73509934])}
'''

print('평균 점수 = ', score['test_score'].mean())
#평균 점수 =  0.7535988149180899

# 단계5. 중요변수 시각화 
X_size = X.shape[1]
plt.barh(range(X_size ), model.feature_importances_) #(y,x)
plt.yticks(range(X_size ), feature_names)
plt.xlabel('importance')
plt.show()
