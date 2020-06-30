'''
 문) digits 데이터 셋을 이용하여 다음과 단계로 Pipeline 모델을 생성하시오.
  <단계1> dataset load
  <단계2> Pipeline model 생성
          - scaling : StndardScaler 클래스, modeing : SVC 클래스    
  <단계3> GridSearch model 생성
          - params : 10e-2 ~ 10e+2, 평가방법 : accuracy, 교차검정 : 5겹
          - CPU 코어 수 : 2개 
  <단계4> best score, best params 출력 
'''

from sklearn.datasets import load_digits # dataset 
from sklearn.svm import SVC # model
from sklearn.model_selection import GridSearchCV # gride search model
from sklearn.pipeline import Pipeline # pipeline
from sklearn.preprocessing import StandardScaler # dataset scaling
from sklearn.model_selection import train_test_split  #split 

# 1. dataset load
digits = load_digits()
X, y = digits.data, digits.target
#data확인
X.min()
X.max()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# 3) SVM model 생성 
svc = SVC(gamma='auto') # 비선형 SVM model 
model = svc.fit(x_train, y_train)

# 4) model평가
score = model.score(x_test, y_test)
score # 0.32407407407407407

# 3. gride search model 
from sklearn.model_selection import GridSearchCV

# [Pipeline]

# 1) pipeline step : [ (step1:scaler),(step2:model),....]
pipe_svc = Pipeline([('scaler',MinMaxScaler()),('svc',SVC(gamma='auto'))])

# 2) pipeline model 
model = pipe_svc.fit(x_train, y_train)

# 3) pipeline model test
score = model.score(x_test, y_test)  

# [ Grid Search : model tuning]

#1) params설정 
params =[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

#dict 형식= {'object__C': params_range) 
params_grid = [{'svc__C':params,'svc__kernel':['linear']}, #선형 
               {'svc__C':params,'svc__gamma':params,'svc__kernel':['rbf']}] #비선형

# 4. best score, best params

gs = GridSearchCV(estimator = pipe_svc, param_grid = params_grid, scoring='accuracy',cv=5, n_jobs=2) 
gs_model = gs.fit(X, y)

#교차검정의 결과 
gs_model.cv_results_["mean_test_score"]






acc = gs_model.score(x_test, y_test)
acc  #0.975925925925926

best_score = gs_model.score(X,y) #0.10127991096271564
best_params = gs_model.best_params_ #{'svc__C': 10.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'}

