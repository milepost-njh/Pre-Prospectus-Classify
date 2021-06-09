#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:42:31 2021

@author: nijiahui
"""


import pandas as pd

data = pd.read_csv('test.csv',encoding='GBK',header=None)
data.columns = ['text', 'label',"pre"]
data = data.drop(["pre"],axis=1)

data_shuffle = data.sample(frac=1).reset_index(drop=True)

data_shuffle.to_csv("./v7_correct_data.csv",index=False)

