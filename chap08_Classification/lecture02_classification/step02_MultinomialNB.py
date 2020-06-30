# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:07:36 2020

@author: user
"""


from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all') # 'train', 'test'
# Downloading 20news dataset.