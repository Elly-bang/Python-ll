# -*- coding: utf-8 -*-
"""
시계열 분석(time series analysis)
 1. 시계열 자료 생성 
 2. 날짜형식 수정(다국어)
 3. 시계열 시각화 
 4. 이동평균 기능 : 5, 10, 20일 평균 ->  추세선 평활(스뮤딩)
"""

from datetime import datetime  # 날짜형식 수정 
import pandas as pd # csv file read
import matplotlib.pyplot as plt # 시계열 시각화 
import numpy as np # 수치 자료  생성 


# 1. 시계열 자료 생성 
time_data = pd.date_range("2017-03-01", "2020-03-30")
time_data # length=1126, freq='D'

# 월 단위 시계열 자료 
time_data2 = pd.date_range("2017-03-01", "2020-03-30", freq = 'M')
time_data2 # freq='M'
len(time_data2) # 36

# 월 단위 매출현황 
x = pd.Series(np.random.uniform(10, 100, size = 36))

df = pd.DataFrame({'date' : time_data2, 'price' : x})
df

plt.plot(df['date'], df['price'], 'g--') # (x, y)
plt.show()


# 2. 날짜형식 수정(다국어)
cospi = pd.read_csv('C:/ITWILL/4_Python-ll/data/cospi.csv')
cospi.info()

cospi.head()
'''
        Date     Open     High      Low    Close  Volume
0  26-Feb-16  1180000  1187000  1172000  1172000  176906
1  25-Feb-16  1172000  1187000  1172000  1179000  128321
'''

date = cospi['Date']
len(date) # 247

# list + for  : 26-Feb-16 -> 2016-02-16
kdate = [ datetime.strptime(d, '%d-%b-%y') for d in date]
kdate

# 날짜 칼럼 수정 
cospi['Date'] = kdate
cospi.info() #datetime64[ns]
cospi.head() # 2016-02-26
cospi.tail() # 2015-03-02


# 3. 시계열 시각화
cospi.index # RangeIndex(start=0, stop=247, step=1)

# 칼럼 -> index 적용 
new_cospi = cospi.set_index('Date')
new_cospi.index 

new_cospi['2016']
len(new_cospi['2015']) # 210
new_cospi['2015']
new_cospi['2015-05':'2015-03']

# subset
new_cospi_HL=new_cospi[['High', 'Low']]
new_cospi_HL.index

#2015년 기준
new_cospi_HL['2015'].plot(title = '2015 year :High vs Low')
plt.show()

#2016년 기준
new_cospi_HL['2016'].plot(title = '2016 year :High vs Low')
plt.show()
#2016년 2월 기준
new_cospi_HL['2016-02'].plot(title = '2016 year :High vs Low')
plt.show()

#4. 이동평균 기능 : 5, 10, 20일 평균 -> 추세선 평활(스무)

#1) 5일 단위 이동 평균 : 5일 단위 평균 -> 마지막 5일째 이동 
roll_mean5= pd.Series.rolling(new_cospi_HL.High, window = 5, center=False).mean()
roll_mean5

# 2) 10일 단위 이동 평균 : 10일 단위 평균 -> 마지박 10일째 이동 
roll_mean10=pd.Series.rolling(new_cospi_HL.High, window = 10, center=False).mean()
roll_mean10

# 2)20일단위 이동 평균 : 20일 단위 평균 -> 마지박 20일째 이동 
roll_mean20= pd.Series.rolling(new_cospi_HL.High, window = 20, center=False).mean()
roll_mean20[:25]

#rolling mean 시각화 
new_cospi_HL.High.plot(color='b', label='high column')
roll_mean5.plot(color='red', label='rolling mean 5day')
roll_mean10.plot(color='green', label='rolling mean 10day')
roll_mean20.plot(color='orange', label='rolling mean 20day')
plt.legend(loc = 'best')
plt.show


 












