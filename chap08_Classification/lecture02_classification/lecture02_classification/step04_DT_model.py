# -*- coding: utf-8 -*-
"""
Decision Tree 모델 
 - 중요변수 선정 기준 : GINI, Entropy
 - GINI : 불확실성을 개선하는데 기여하는 척도 
 - Entropy : 불확실성을 나타내는 척도 
"""

from sklearn.model_selection import train_test_split # split
from sklearn.datasets import load_iris, load_wine # dataset 
from sklearn.tree import DecisionTreeClassifier, export_graphviz # tree model 
from sklearn.metrics import accuracy_score, confusion_matrix # model평가 

# tree 시각화 관련 
from sklearn.tree.export import export_text # tree 구조 텍스트 
from sklearn import tree


iris = load_iris()
names = iris.feature_names
names 
iris.target_names # ['setosa', 'versicolor', 'virginica']

X = iris.data 
y = iris.target

X.shape #  (150, 4)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3)

help(DecisionTreeClassifier)

dtc = DecisionTreeClassifier(criterion='gini', 
                             random_state=123,
                             max_depth=3)

dtc2 = DecisionTreeClassifier(criterion='entropy')

model = dtc.fit(X=x_train, y=y_train)
model2 = dtc2.fit(X=x_train, y=y_train)

# tree  시각화 
tree.plot_tree(model)
tree.plot_tree(model2)

print(export_text(model, names))
'''
|--- petal length (cm) <= 2.45  : 3번 칼럼 분류조건(왼쪽 노드) 
|   |--- class: 0   -> 'setosa' 100% 분류 
|--- petal length (cm) >  2.45  : 3번 칼럼 분류조건(오른쪽 노드)
|   |--- petal length (cm) <= 4.85
|   |   |--- petal width (cm) <= 1.70
|   |   |   |--- class: 1
|   |   |--- petal width (cm) >  1.70
'''

y_pred = model.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc # 0.9111111111111111 -> max_depth=3

confusion_matrix(y_true, y_pred)
'''
array([[19,  0,  0],
       [ 0, 11,  2],
       [ 0,  2, 11]], dtype=int64)
'''

y_pred = model2.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc # 0.9555555555555556 -> max_depth=6

confusion_matrix(y_true, y_pred)


###########################
## model overfitting
###########################

wine = load_wine()
x_name = wine.feature_names # x변수명 
X = wine.data
y = wine.target


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=123)


dt = DecisionTreeClassifier() # default 
model = dt.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
train_score # 1.0
test_score = model.score(x_test, y_test)
test_score # 0.9259259259259259

tree.plot_tree(model) # max_depth=5


# max_depth=3
dt = DecisionTreeClassifier(max_depth=3)  
model = dt.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
train_score # 0.9838709677419355
test_score = model.score(x_test, y_test)
test_score # 0.9629629629629629

tree.plot_tree(model) # max_depth=3

export_graphviz(model, out_file='DT_tree_graph.dot',
                feature_names=x_name, max_depth=3,
                class_names=True)

















