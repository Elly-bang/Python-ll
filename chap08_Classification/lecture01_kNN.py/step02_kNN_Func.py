# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:42:16 2020

C:\ITWILL\4_Python-ll\workspace\chap08_Classification
"""
#from module import function 
from step01_kNN_data import data_set
import numpy as np

#dataser 생성
know,not_know, cate = data_set()

know.shape #(4, 2)
know
'''
array([[1.2, 1.1],
       [1. , 1. ],
       [1.8, 0.8],
       [2. , 0.9]])
'''
not_know.shape #(2,)
cate #array(['A', 'A', 'B', 'B'], dtype='<U1')

#거리계산식 : 차 > 제곱 > 합 > 제곱근
diff = know - not_know
diff

'''
array([[-0.4 ,  0.25],
       [-0.6 ,  0.15],
       [ 0.2 , -0.05],
       [ 0.4 ,  0.05]])
'''
aquare_diff = diff**2
aquare_diff
'''
array([[0.16  , 0.0625],
       [0.36  , 0.0225],
       [0.04  , 0.0025],
       [0.16  , 0.0025]])
'''

sum_aquare_diff= aquare_diff.sum(axis = 1)
sum_aquare_diff #array([0.2225, 0.3825, 0.0425, 0.1625])

distance = np.sqrt(sum_aquare_diff)
distance #array([0.47169906, 0.61846584, 0.20615528, 0.40311289])

sortDist = distance.argsort()
sortDist #[2, 3, 0, 1]

result = cate[sortDist] 
result #['B', 'B', 'A', 'A']

#k =3 최근접 이웃 3개 
k3 = result[:3] #['B', 'B', 'A']

#dict 
classify_re = {}
for key in k3 :
    classify_re[key] =  classify_re.get(key,0) + 1
    
classify_re #{'B': 2, 'A': 1}

vote_re = max(classify_re)
print('분류 결과 : ', vote_re) #분류 결과 :  B


def knn_classify(know, not_know, cate, k ) :  #k=3실행수=>class_result = knn_classify(know, not_know, cate )
    #유클리드 거리계산식
    
    # [단계1] 거리계산식 : 차 > 제곱 > 합 > 제곱근
    diff = know - not_know
    aquare_diff = diff**2  
    sum_aquare_diff= aquare_diff.sum(axis = 1)
    distance = np.sqrt(sum_aquare_diff)
    
    #[단계2] 오름차순 정렬 ->index
    sortDist = distance.argsort()   

    #[단계3] 최근접 이웃(k=3) 
    class_result = {}  #빈set 
    
    for i in range(k) : #k=3_  (0~2) 3회 반복
        key = cate[sortDist[i]] #최근접 이웃을 꺼내서 시작한다.
        class_result[key] = class_result.get(key,0) + 1

    return class_result


class_result = knn_classify(know, not_know, cate, 3 )

class_result  #{'B': 2, 'A': 1}

print('분류 결과 : ', max(class_result)) #분류 결과 :  B

































