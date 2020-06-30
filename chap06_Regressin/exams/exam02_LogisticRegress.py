'''
문) load_wine() 함수를 이용하여 와인 데이터를 다항분류하는 로지스틱 회귀모델을 생성하시오. 
  조건1> train/test - 7:3비울
  조건2> y 변수 : wine.data 
  조건3> x 변수 : wine.data
  조건4> 모델 평가 : confusion_matrix, 분류정확도[accuracy]
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# 1. wine 데이터셋 
wine = load_wine()
wine.target_names #['class_0', 'class_1', 'class_2']

# 2. 변수 선택 
wine_x = wine.data # x변수 
wine_y = wine.target # y변수

# 3. train/test split(7:3)
x_train, x_test, y_train, y_test = train_test_split(wine_x, wine_y, test_size=0.3) 

# 4. model 생성  : solver='lbfgs', multi_class='multinomial' 일반 데이터, 다항분류 
lr = LogisticRegression(random_state = 123, solver='lbfgs', multi_class='multinomial',n_jobs=1, verbose=1)

#학습과정 출력 여부 :verbose=1
#병렬처리 cpu수 : n_jobs=1

model = lr.fit(x_train , y_train)
model 

# 5. 모델 평가 : accuracy, confusion matrix
y_pred = model.predict(x_test)
acc = accuracy_score(x_test, y_test) 
print('accuracy = ', acc)  # 0.9733333333333334

;;;
acc= metrics.accuracy_score(y_test, y_pred)
acc # 0.9629629629629629


y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy = ', acc)


con_mat =confusion_matrix(y_test, y_pred)
print(con_mat)

'''
[[16  1  0]
 [ 0 24  1]
 [ 0  0 12]]
'''

acc = (16+24+12) / con.mat.sum()
print('accuracy = ', acc)   #0.9629629629629629

################################
###digit : multi class 
################################
from sklearn.datasets import load_digits

#1. dataset load
digits = load_digits()
digits.target_names #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X = digits.data 
y = digits.target
X.shape # (1797, 64)
y.shape #1797장 images 1진수 정답

#2.split
img_train, img_test, label_train,label_test = train_test_split(X, y, test_size = 0.25)

import matplotlib.pyplot as plt

img2d = img_train.reshape(-1,8,8) #(전체 image, 세로, 가로)
img2d.shape #1347, 8, 8
img2d[0]
plt.imshow(img2d[0])
img_train[0]
label_train[0]

img_test.shape


#3.model생성

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
model = lr.fit(img_train, label_train)
y_pred = model.predict(img_test)

# 4. model평가 
acc = accuracy_score(label_test, y_pred)
print(acc)

con_mat = confusion_matrix(label_test,y_pred)
con_mat 

result = label_test == y_pred
result
len(result)




























































