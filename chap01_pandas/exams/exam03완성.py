'''  
step02 관련문제
문3) 다음 df를 대상으로 iloc 속성을 이용하여 단계별로 행과 열을 선택하시오.
   단계1 : 1,3행 전체 선택    
   단계2 : 1~4열 전체 선택 
   단계3 : 1,3행 1,3,5열 선택
'''
import pandas as pd
import numpy as np

data = np.arange(1, 16).reshape(3,5) # 3x5

df = pd.DataFrame(data, 
            index = ['one', 'two', 'three'],
            columns = [1,2,3,4,5])
print(df)
'''
        1   2   3   4   5
one     1   2   3   4   5
two     6   7   8   9  10
three  11  12  13  14  15
'''

# 단계1 : 1,3행 전체 선택    
df.iloc[[0, 2]]
df.loc[['one','three']]

# 단계2 : 1~4열 전체 선택 
df.iloc[:,:4]
df.loc[:,1:4]

# 단계3 : 1,3행 1,3,5열 선택
df.iloc[[0,2], [0,2,4]]
df.loc[['one','three'], [1,3,5]]



   

