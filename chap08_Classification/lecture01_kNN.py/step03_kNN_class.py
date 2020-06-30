# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:39:00 2020
"""

#from module import function 
from step01_kNN_data import data_set
import numpy as np

#dataser 생성
know,not_know, cate = data_set()

class kNNClassify : 
     #생성자, 멤버(메서드, 변수)
    
    def classify(self, know, not_know, cate, k=3) : 
         # [단계1] 거리계산식 : 차 > 제곱 > 합 > 제곱근
        diff = know - not_know
        aquare_diff = diff**2  
        sum_aquare_diff= aquare_diff.sum(axis = 1)
        distance = np.sqrt(sum_aquare_diff)
        
        #[단계2] 오름차순 정렬 ->index
        sortDist = distance.argsort()   
    
        #[단계3] 최근접 이웃(k=3) 
        self.class_result = {}  #멤버 변수 
        
        for i in range(k) :
            key = cate[sortDist[i]] 
            self.class_result[key] = self.class_result.get(key,0) + 1
    
    def vote(self) :
         vote_re = max(self.class_result)
         print('분류 결과 : ', vote_re)

#객체 생성 : 생성자 이용 
knn = kNNClassify()
knn.classify(know, not_know, cate)    
knn.class_result # {'B': 2, 'A': 1}
knn.vote()    #분류 결과 :  B
