# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:38:00 2020

@author: user
- 1차원 indexing :list동일함 
- 2,3차 indexing 
 -boolean indexing  
"""

import numpy as np

#1. indexing 
'''
1차원 : obj [index]
2차원 : obj [행index, 열 index]
3차원 : obj [면index, 행index, 열 index]
'''

#1)list객체 
ldata =[0,1,2,3,4,5]
ldata
ldata[:] #전체원소
ldata[2:] #[n:~]
ldata[:3] #[~:n]
ldata[-1] 

#2)numpy 객체 
arrld = np.array(ldata)
arrld.shape  #(6,)
arrld[2:]
arrld[:3] #array([0, 1, 2])
arrld[-1] #5

#2. slicing 
arr= np.array(range(1,11))
arr #원본 array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr_sl =arr[4:8]
arr_sl #사본 array([5, 6, 7, 8])

#블럭 수정
arr_sl[:]= 50 
arr_sl #array([50, 50, 50, 50])

arr #array([ 1,  2,  3,  4, 50, 50, 50, 50,  9, 10]) 원본도 수정된다. 

# 2,3 차 indexing 
arr2d = np.array([[1,2,3],[2,3,4],[3,4,5]])
arr2d
'''
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]])
'''
arr2d.shape
#(3, 3)

#행 index : default 
arr2d[1] #arr2d[1, :] -> [2, 3, 4]
arr2d[1:] #2~3행 
'''
array([[2, 3, 4],
       [3, 4, 5]])
'''

arr2d[:,1:] #2~3열
'''array([[2, 3],
       [3, 4],
       [4, 5]])'''
arr2d[2,1] #3행 2열 
arr2d[:2,1:] #box선택 
'''
array([[2, 3],
       [3, 4]])
'''


#[면, 행, 열] 면 index : default 
arr3d= np.array([ [[1,2,3],[2,3,4,],[3,4,5]],
                  [[2,3,4],[3, 4,5],[6,7,8]] ]) #2,3 

arr3d
'''
Out[163]: 
array([[[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]],

       [[2, 3, 4],
        [3, 4, 5],
        [6, 7, 8]]])

array([[2, 3],
       [3, 4],
       [4, 5]])

'''


arr3d[0] #면 index = 1면 
arr3d[1] 


#면- > 행 index = 1면
arr3d[0,2] #3,4,5

#면 -> 행-> 열 index
arr3d[1,2,2] #8
arr3d[1,1:,1:]

#4. boolean indexing
dataset = np.random.randint(1,10,size=100) #1~10
len(dataset) #100

#5이상
dataset
dataset2=dataset[dataset >= 5]
len(dataset2) #52
dataset2

#5~8 자료 선택 
dataset[dataset>= 5 and dataset <=8] #error 
np.logical_and
np.logical_or
np.logical_not
dataset2 = dataset[np.logical_and(dataset >= 5, dataset <=8)]
dataset2






















