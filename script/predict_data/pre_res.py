#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 16:16:15 2021

@author: nijiahui
"""


import pandas as pd 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = "data/example_data.csv"
model_path = "model/fashion_mnist_model.h5"
pickle_path = "pickle_file/dict_word.pkl"

model = load_model(model_path)
with open(pickle_path,"rb") as f:
    tokenizer = pickle.load(f)
    
data = pd.read_csv(data_path)

x_data = tokenizer.texts_to_sequences(data["name"])


x_data = pad_sequences(x_data, padding='post', truncating='post')
y_pred = model.predict(x_data)
pre_res = y_pred.argmax(-1)

data["pre"] = pre_res


data.to_csv("pre_res.csv")
