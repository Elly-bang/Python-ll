# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:34:09 2020

@author: user
"""

Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 7.12.0 -- An enhanced Interactive Python.

lst=[1,2,3]

lst2=[]
for i in lst:
    print(i**2)

arr = np.array(lst)
arr
arr**2 

arr = np.arry([1,'two',3])
arr
1
4
9
Traceback (most recent call last):

  File "<ipython-input-1-de71a8c9030b>", line 7, in <module>
    arr = np.array(lst)

NameError: name 'np' is not defined


arr = np.array(lst)
Traceback (most recent call last):

  File "<ipython-input-2-9106dac92f5c>", line 1, in <module>
    arr = np.array(lst)

NameError: name 'np' is not defined


arr = np.array(lst)
Traceback (most recent call last):

  File "<ipython-input-3-9106dac92f5c>", line 1, in <module>
    arr = np.array(lst)

NameError: name 'np' is not defined


arr
Traceback (most recent call last):

  File "<ipython-input-4-24a6d41c5b66>", line 1, in <module>
    arr

NameError: name 'arr' is not defined


arr**2 
Traceback (most recent call last):

  File "<ipython-input-5-93d0a57d8b24>", line 1, in <module>
    arr**2

NameError: name 'arr' is not defined


arr = np.arry([1,'two',3])
Traceback (most recent call last):

  File "<ipython-input-6-0e4a00a9c96e>", line 1, in <module>
    arr = np.arry([1,'two',3])

NameError: name 'np' is not defined


arr
Traceback (most recent call last):

  File "<ipython-input-7-24a6d41c5b66>", line 1, in <module>
    arr

NameError: name 'arr' is not defined



import numpy as np

lst=[1,2,3]

for i in lst:
    print(i**2)

arr = np.array(lst)
arr
arr**2 

arr = np.arry([1,'two',3])
arr
1
4
9
Traceback (most recent call last):

  File "<ipython-input-8-b3566ca72dc1>", line 12, in <module>
    arr = np.arry([1,'two',3])

  File "C:\Users\user\anaconda3\lib\site-packages\numpy\__init__.py", line 220, in __getattr__
    "{!r}".format(__name__, attr))

AttributeError: module 'numpy' has no attribute 'arry'


arr = np.array([1,'two',3])

arr
Out[10]: array(['1', 'two', '3'], dtype='<U11')

arr.shape
Out[11]: (3,)

arr = np.array([[1,'two',3]])

arr #array(['1', 'two', '3'], dtype='<U11')
Out[13]: array([['1', 'two', '3']], dtype='<U11')

arr.shape # (3,)
Out[14]: (1, 3)


#1. random : 난수 생성 함수 
data = np.random.randn(3, 4)
data
'''
array([[ 0.21743644, -0.38840953, -0.099379  ,  1.71704387],
       [ 1.69258254, -0.05811048,  1.0259971 , -1.56414025],
       [ 1.25136511,  0.29515078, -1.03241856, -0.49303029]])
'''

for row in data :
    print('행 단위 합계 :', row.sum())
    print('행 단위 평균 :', row.mean())
    
# 1) 수학/통계 함수 지원
type(data) #numpy.ndarray
print('전체 합계: ', data.sum())
print('전체 평균: ', data.mean())
print('전체 분산: ', data.var())
print('전체 표준편차: ', data.std())

dir(data) #멤버 확인, 개체 확인 
data.shape #(3, 4)
data.size #12

#2)범위 수정, 블럭 연산
data + data 
'''
array([[ 0.43487288, -0.77681907, -0.198758  ,  3.43408775],
       [ 3.38516507, -0.11622095,  2.0519942 , -3.1282805 ],
       [ 2.50273021,  0.59030156, -2.06483712, -0.98606058]])
'''
data - data 
'''
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
'''
#3) indexing : R과 유사
data[0,0] #1행 1열 
data[0,:] #1행 전체
data[:,1] #2열 전체 
 
#2. array 함수 N차원 배열 생성
#1) 단일 list 
lst1 = [3, 5.6, 4, 7, 8]
lst1
lst1.var() 

#list -> array

arr1 = np.array(lst1)
arr1  #array([3. , 5.6, 4. , 7. , 8. ])

arr1.var()
arr1.std()

#2)중첩 list
lst2 = [[1,2,3,4,5],[2,3,4,5,6]]
lst2 # [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
lst2[0][2] #3

arr2 = np.array(lst2)
arr2
#array([[1, 2, 3
, 4, 5], -1행 (0)
#       [2, 3, 4, 5, 6]]) -2행(1)
    
arr2.shape #(2,5)
arr2[0,2]  #3

#index :obj[행index, 열 index]
arr2[1,:] #array([2, 3, 4, 5, 6])

arr2[:,1] #2열 전체  array([2, 3])
arr2[:, 2:4]
'''
array([[3, 4],
       [4, 5]])
'''

#broadcast연산
#- 작은 차원이 큰 차원으로 늘어난 후 연산
#1) scala(0) vs vector(1)
0.5*arr1 # array([1.5, 2.8, 2. , 3.5, 4. ])
#  [3, 5.6, 4, 7, 8]
#1) scala(1) vs matrix(2)
0.5* arr2
'''
array([[0.5, 1. , 1.5, 2. , 2.5],
       [1. , 1.5, 2. , 2.5, 3. ]])
'''

# 3) vector(1) vs matrix(2)
print(arr1.shape, arr2.shape)
#(5,) (2, 5)

arr3 = arr1 + arr2 
arr3

# 3. sampling 함수 
num=list(range(1,11))
num # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
help(np.random.choice)
#(a, size=None, replace=True, p=None)
'''
a = 관측치 길이
size = 임의 추출 크기 
replace  = 복원(True) or 비복원(False)
p = 확률 
'''

idx = np.random.choice(a=len(num), size=5, replace=False ) #비복원 추출 

idx  #array([6, 2, 1, 9, 5])

import pandas as pd
score=pd.read_csv('C:/ITWILL/4_Python-ll/data/data/score_iq.csv')
score.info()
len (score) #150


idx = np.random.choice(a=len(score), size=int(len(score)*0.3), replace=False )

idx  #array([104,  81,  93,  73,  35,  43,  96,  30,  17, 129,  68, 126,  40,
         4,  71,  11,  19, 124, 118, 105,  10,   3, 149,  49,  63,  38,
       137, 135,  66, 140,  20,  44,   7, 141,  72,  87, 132,  16,  14,
       113, 107,  91,   5,  67, 103])


len(idx) #45

#DataFrame index 
score_train = score.iloc[idx,:]
score_train
score_train.shape #(45, 6)

#pandas(DF) -> numpy(array)
score_arr = np.array(score)
score_arr.shape #(150, 6)
score_train2 = score_arr[idx,:]
score_train2.shape #(45, 6)

#test set 선택
test_idx= [ i for i in range(len(score))]





#4.arrange함수 : range()유사
zero_arr = np.zeros((3,5))
zero_arr

cnt = 1 
for i in range(3) : #행index
    for j in range(5):#열index
        zero_arr[i,j] = cnt 
        cnt += 1 
        
zero_arr
'''
array([[ 1.,  2.,  3.,  4.,  5.],
       [ 6.,  7.,  8.,  9., 10.],
       [11., 12., 13., 14., 15.]])'''

cnt = 1 
for i in np.arange(3) : #행index
    for j in np.arange(5):#열index
        zero_arr[i,j] = cnt 
        cnt += 1 
zero_arr
'''
array([[ 1.,  2.,  3.,  4.,  5.],
       [ 6.,  7.,  8.,  9., 10.],
       [11., 12., 13., 14., 15.]])
'''

range(-1.0, 2, 0.1) #(start,stop,setp)
np.arange(-1.0, 2, 0.1)

'''
array([-1.00000000e+00, -9.00000000e-01, -8.00000000e-01, -7.00000000e-01,
       -6.00000000e-01, -5.00000000e-01, -4.00000000e-01, -3.00000000e-01,
       -2.00000000e-01, -1.00000000e-01, -2.22044605e-16,  1.00000000e-01,
        2.00000000e-01,  3.00000000e-01,  4.00000000e-01,  5.00000000e-01,
        6.00000000e-01,  7.00000000e-01,  8.00000000e-01,  9.00000000e-01,
        1.00000000e+00,  1.10000000e+00,  1.20000000e+00,  1.30000000e+00,
        1.40000000e+00,  1.50000000e+00,  1.60000000e+00,  1.70000000e+00,
        1.80000000e+00,  1.90000000e+00])
'''
    
    


