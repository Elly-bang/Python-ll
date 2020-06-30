'''
문1) iris.csv 파일을 이용하여 다음과 같이 차트를 그리시오.
    <조건1> iris.csv 파일을 iris 변수명으로 가져온 후 파일 정보 보기
    <조건2> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그리기
    <조건3> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그린 후  5번 칼럼으로 색상 적용 
'''
# plt.scatter(1번 칼럼, 3번 칼럼, c=5번 칼럼, marker='o')

import pandas as pd
import matplotlib.pyplot as plt

# 조건1> 
iris = pd.read_csv('iris.csv')
iris.info()

# 조건2> 
plt.scatter(iris['Sepal.Length'], iris['Petal.Length'])

# 조건3>
species = iris['Species']
species.unique()
# ['setosa', 'versicolor', 'virginica']

len(species) # 150

cdata = [] # 빈 list
for s in species :
    if s == 'setosa' :
        cdata.append(3)
    elif s == 'versicolor' :
        cdata.append(4)
    else :
        cdata.append(5)


plt.scatter(iris['Sepal.Length'],
            iris['Petal.Length'],
            c = cdata, marker='o')







