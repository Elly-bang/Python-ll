# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:17:24 2020

@author: user

reshape vs resize

- reshape : 모양 변경
- resize : 크키 변경 
 ex) images -> 120 X 150 규격화 -> model

image 규격화 : 실습
"""
from glob import glob #file 검색 패턴 사용 (문자열 경로, *.jpg)
from PIL import Image #image file read  
import numpy as np
import matplotlib.pyplot as plt #이미지 시각화

#1개 image file open 
path = "./chap04_NumPy"
file =  path + '/images/test1.jpg'

img = Image.open(file) #image file read 
type(img) #PIL.JpegImagePlugin.JpegImageFile
img.shape #shape사용 못함 type(img)  
np.shape(img) # (360, 540, 3) -> (120,150,3) -> (h, w, c)
plt.imshow(img)

img_re = img.resize((150,120)) #원하는 사이즈의 가로 세로 사이즈를 바꿔주어야 함
np.shape(img_re)
plt.imshow(img_re)

#PIL ->numpy
type(img_re) #PIL.Image.Image
img_arr = np.asarray(img)
img_arr.shape # (360, 540, 3)
type(img_arr) #numpy.ndarray

#여러장의 image resize함수 
def imageResize():
    img_h =120
    img_w =150
    
    image_resize = [] #규격화된 image 저장 
    
    #glob : file 패턴
    for file in glob(path + '/images/' + '*.jpg') :
        #test1.jpg -> test.jpg        , ...
        img = Image.open(file) #image file open
        print(np.shape(img)) #image shape 
    
        
        #PIL -> resize
        img = img.resize(( img_w, img_h)) #W, h 
        #PIL -> numpy
        img_data=np.asarray(img)

        #resize image save
        image_resize.append(img_data)
        print(file, ':', img_data.shape)
      
        #list -> numpy 
        
    return np.array(image_resize)
 

image_resize= imageResize()
"""
(360, 540, 3)
./chap04_NumPy/images\test1.jpg : (360, 540, 3)
(332, 250, 3)
./chap04_NumPy/images\test2.jpg : (332, 250, 3)
   image_resize,shape
"""
 
image_resize[0].shape #(120, 150, 3)
image_resize[1].shape # (120, 150, 3)

# image 보기 
plt.imshow(image_resize[0])
plt.imshow(image_resize[1])
  
    











