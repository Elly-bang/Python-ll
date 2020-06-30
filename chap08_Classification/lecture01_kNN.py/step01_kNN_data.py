# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:48:39 2020

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:12:17 2020

@author: user
"""
import numpy as np # 다차원배열, 선형대수 연산 
import matplotlib.pyplot as plt

# 1. 알려진 두 집단 x,y 산점도 시각화 
plt.scatter(1.2, 1.1) # A 집단
plt.scatter(1.0, 1.0)
plt.scatter(1.8, 0.8) # B 집단 
plt.scatter(2, 0.9)

plt.scatter(1.6, 0.85, color='r') # 분류대상 
plt.show()

#분류대상을 A,B집단으로부터 각 scatter의 거리를 확인 , 가장 가까운 상위 3개를 통해 어느 집단에 속하는지 분류

# 2. DATA 생성(1차원의 리스트)과 함수 정의 
p1 = [1.2, 1.1] # A 집단 
p2 = [1.0, 1.0]
p3 = [1.8, 0.8] # B 집단
p4 = [2, 0.9]
category = ['A','A','B','B'] # 알려진 집단 분류범주(Y변수)
p5 = [1.6, 0.85] # 분류대상 

# data 생성 함수 정의
def data_set():
    # 선형대수 연산 : numpy형 변환 
    know_group = np.array([p1, p2, p3, p4]) # 알려진 집단 - 중첩list 2차원
    not_know_group = np.array(p5) # 알려지지 않은 집단 - 1차원 #차원이 달라도 계산이 됩니다 :broadcast 
    class_category = np.array(category) # 정답(분류범주)
    return know_group,not_know_group,class_category 

know,not_know, cate = data_set()
know.shape #(4, 2)
not_know.shape #(2,)
cate.shape #(4,)
